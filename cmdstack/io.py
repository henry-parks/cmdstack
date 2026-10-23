"""Read one or more CSV (or Excel) exports and count each distinct command line, without loading CSVs into memory."""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import duckdb
import pandas as pd

# column names used by common EDR, SIEM, Sysmon and forensic exports, tried in order when no option names one
COMMAND_LINE_CANDIDATES = [
    "CommandLine", "command_line", "commandline", "ProcessCommandLine", "process_command_line",
    "process.command_line", "Process_Command_Line", "TargetProcessCommandLine", "tgt.process.cmdline",
    "cmdline", "cmd_line", "Command", "ProcessCmdLine", "ExecutableInfo",
]
HOST_CANDIDATES = [
    "DeviceName", "Computer", "ComputerName", "host", "host.name", "hostname", "Host_Name", "endpoint.name",
    "DvcHostname", "agent.hostname", "aid",
]   # not "dest": it is a host in some SIEM endpoint data models but an address in network data; pass --host-column dest for it
COUNT_CANDIDATES = ["count", "count_", "events", "event_count", "EventCount"]
HOSTS_COUNT_CANDIDATES = ["hosts", "host_count", "HostCount", "dcount_DeviceName", "dcount_Computer"]
PARENT_CANDIDATES = [
    "ParentProcessName", "InitiatingProcessFileName", "ParentImage", "CreatorProcessName", "process.parent.name",
    "ParentBaseFileName", "src.process.name", "parent",
]
EXCEL_SUFFIXES = (".xlsx", ".xlsm")
CSV_OPTIONS = "all_varchar=true, header=true, ignore_errors=true, max_line_size=10000000"
LENIENT = ", delim=',', quote='\"', escape='\"', strict_mode=false"   # also tolerates mixed line endings


HOST_SAMPLE = 200  # host names kept per line, for scoping; a group seen on more hosts is reported as "200+"


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _columns(con, source: str) -> list[str]:
    return [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()]


def find_column(columns: list[str], wanted: str | None, candidates: list[str], what: str) -> str | None:
    """Match a column name case-insensitively; with no explicit name, try the common candidates.

    wanted="none" turns the column off (no detection).
    """
    lower = {c.lower(): c for c in columns}
    if wanted and wanted.lower() == "none":
        return None
    if wanted:
        if wanted.lower() not in lower:
            raise SystemExit(f"cmdstack: no {what} column named {wanted!r}. Columns found: {', '.join(columns)}")
        return lower[wanted.lower()]
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def read_events(paths, column=None, count=None, host=None, label=None, parent=None,
                hosts_count=None) -> tuple[pd.DataFrame, dict]:
    """Return one row per distinct command line with its event count and optional context columns.

    Columns: command_line, events, and when the matching input column exists: hosts (distinct count),
    labels (distinct values joined with '|'), parent (one example parent process).

    hosts_count names a column that already holds a number of hosts (an export aggregated in the SIEM, e.g. KQL
    dcount(DeviceName)). It is summed per line and used only when no host-name column is given; a group's host
    count is then a lower bound, because the export cannot say whether two lines ran on the same machines.
    """
    files = [str(Path(p)) for p in paths]
    for f in files:
        if not Path(f).exists():
            raise SystemExit(f"cmdstack: file not found: {f}")
        if f.lower().endswith(".xls"):
            raise SystemExit(f"cmdstack: {f} is an old-format Excel file (.xls), which cmdstack cannot read. "
                             "Open it in Excel and save it as .xlsx or CSV.")
    tmp_dir = Path(tempfile.mkdtemp(prefix="cmdstack-"))
    try:
        return _read(files, _excel_to_csv(files, tmp_dir), column, count, host, label, parent, hosts_count)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)   # converted Excel sheets hold the same sensitive data


def _sql_path(f: str) -> str:
    return "'" + f.replace("'", "''") + "'"


def _skipped_rows(con, files: list[str], options: str) -> int:
    """Rows the reader could not parse (for example too many fields), which ignore_errors would drop silently.

    DuckDB records rejects only per file and only when every column is parsed, hence one empty full scan per file.
    """
    skipped = 0
    for f in files:
        con.execute(f"CREATE OR REPLACE TEMP TABLE _scan AS SELECT * FROM read_csv({_sql_path(f)}, {options}, "
                    f"store_rejects=true) WHERE random() < 0")
        skipped += con.execute("SELECT count(DISTINCT (file_id, line)) FROM reject_errors").fetchone()[0]
    return skipped


def _csv_options(con, files: list[str]) -> str:
    """DuckDB guesses the CSV dialect from a sample, and the guess fails when a sampled row has extra fields; strict
    parsing also fails on mixed line endings (exports pasted together). SIEM and EDR consoles write standard
    comma-separated CSV, so fall back to stating that, non-strict. _read retries the same way if reading fails."""
    try:
        for f in files:
            con.execute(f"DESCRIBE SELECT * FROM read_csv({_sql_path(f)}, {CSV_OPTIONS})")
        return CSV_OPTIONS
    except duckdb.Error:
        return CSV_OPTIONS + LENIENT


def _read(given, files, column, count, host, label, parent, hosts_count, options=None):
    con = duckdb.connect()
    if not sys.stderr.isatty():                  # DuckDB's progress bar floods piped or saved logs
        con.execute("SET enable_progress_bar = false")
    options = options or _csv_options(con, files)
    file_list = "[" + ", ".join(_sql_path(f) for f in files) + "]"
    source = f"read_csv({file_list}, {options}, union_by_name=true)"
    try:
        columns = _columns(con, source)
    except duckdb.Error as e:
        raise SystemExit(f"cmdstack: could not read {', '.join(given)} as CSV: {str(e).splitlines()[0]}\n"
                         "Export it again as comma-separated CSV with a header row, or save it as .xlsx.")

    cmd_col = find_column(columns, column, COMMAND_LINE_CANDIDATES, "command-line")
    if cmd_col is None:
        raise SystemExit("cmdstack: could not find a command-line column; pass --cmd-column. "
                         f"Columns found: {', '.join(columns)}")
    # optional columns: use the one named, or detect a common name (the CLI prints what it found)
    host_col = find_column(columns, host, HOST_CANDIDATES, "host")
    count_col = find_column(columns, count, COUNT_CANDIDATES, "count")
    parent_col = find_column(columns, parent, PARENT_CANDIDATES, "parent")
    hosts_count_col = None if host_col else find_column(columns, hosts_count, HOSTS_COUNT_CANDIDATES, "hosts-count")
    label_col = find_column(columns, label, [], "label") if label else None
    detected = [name for name, col, given in (("host", host_col, host), ("count", count_col, count),
                                               ("parent", parent_col, parent), ("hosts-count", hosts_count_col,
                                                                                hosts_count))
                if col and not given]

    events = f"SUM(TRY_CAST({_quote(count_col)} AS DOUBLE))" if count_col else "COUNT(*)"
    select = [f"{_quote(cmd_col)} AS command_line", f"{events} AS events"]
    if host_col:
        host_value = f"NULLIF(TRIM({_quote(host_col)}), '')"
        valid_host = f"{host_value} IS NOT NULL"
        select.append(f"COUNT(DISTINCT {host_value}) FILTER (WHERE {valid_host}) AS hosts")
        # a few host names per line, so a group's host count can be taken across its lines (see report.HOST_SAMPLE)
        host_list = f"list(DISTINCT {host_value}) FILTER (WHERE {valid_host})"
        select.append(f"list_slice({host_list}, 1, {HOST_SAMPLE}) AS host_sample")
    elif hosts_count_col:
        select.append(f"SUM(TRY_CAST({_quote(hosts_count_col)} AS DOUBLE)) AS hosts")
    if label_col:
        select.append(f"STRING_AGG(DISTINCT {_quote(label_col)}, '|') AS labels")
    if parent_col:
        select.append(f"ANY_VALUE({_quote(parent_col)}) AS parent")
    query = (f"SELECT {', '.join(select)} FROM {source} "
             f"WHERE {_quote(cmd_col)} IS NOT NULL AND TRIM({_quote(cmd_col)}) <> '' "
             f"GROUP BY {_quote(cmd_col)}")
    try:
        frame = con.execute(query).df()
        total_rows = con.execute(f"SELECT COUNT(*) FROM {source}").fetchone()[0]
        skipped = _skipped_rows(con, files, options)
    except duckdb.Error as e:
        if LENIENT in options:
            raise SystemExit(f"cmdstack: could not read {', '.join(given)} as CSV: {str(e).splitlines()[0]}\n"
                             "Export it again as comma-separated CSV with a header row, or save it as .xlsx.")
        con.close()
        return _read(given, files, column, count, host, label, parent, hosts_count, CSV_OPTIONS + LENIENT)
    con.close()
    frame["events"] = frame["events"].fillna(0).round().astype("int64")
    if "hosts" in frame.columns:
        frame["hosts"] = frame["hosts"].fillna(0).round().astype("int64")
    info = {"files": list(given), "rows_read": int(total_rows), "rows_skipped": int(skipped),
            "lenient": LENIENT in options,
            "command_line_column": cmd_col,
            "count_column": count_col, "host_column": host_col, "label_column": label_col,
            "parent_column": parent_col,
            "hosts_count_column": hosts_count_col, "hosts_exact": bool(host_col), "detected": detected,
            "columns": columns}
    return frame, info


def _excel_to_csv(files: list[str], tmp_dir: Path) -> list[str]:
    """Excel exports (some EDR consoles save .xlsx) are converted to CSV files in tmp_dir; the first sheet is read."""
    out = []
    for i, f in enumerate(files):
        if not f.lower().endswith(EXCEL_SUFFIXES):
            out.append(f)
            continue
        frame = pd.read_excel(f, sheet_name=0, dtype=str)
        tmp = tmp_dir / f"{i}-{Path(f).stem}.csv"
        frame.to_csv(tmp, index=False)
        out.append(str(tmp))
    return out
