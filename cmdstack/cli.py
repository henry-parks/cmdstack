"""cmdstack command-line interface."""
from __future__ import annotations

import argparse
import getpass
import glob
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .baseline import (BASELINE_FILE, MATCH_THRESHOLD, REVIEWED_FILE, REVIEWED_NEW, load_baseline, load_reviewed,
                       save_reviewed)
from .io import read_events
from .normalize import PLATFORMS, parse_networks
from .pipeline import DEFAULT_MODEL, default_threads, limit_threads, load_encoder, run
from .report import write_outputs
from .search import search_main

EXAMPLES = """examples:
  cmdstack events.csv                          columns are detected; report in cmdstack-out/report.html
  cmdstack export.xlsx -o results              Excel (.xlsx) exports work too
  cmdstack week.csv --history hunts            weekly use: compares with the last run in hunts/

output (in the run folder): report.html to read, findings.md for a ticket, groups.csv and group_hosts.csv.
Lines too unusual to group with others are listed as ungrouped lines, not dropped.

marking routine groups so later runs hide them:
  cmdstack review hunts --import Downloads\\reviewed.json   the file the report's Mark reviewed button downloads
  cmdstack review hunts --mark g1a2b3c4d5 --note "backup agent"
  cmdstack review hunts --list                               (see: cmdstack review --help)

scoping a finding: every line like it, and the machines it ran on:
  cmdstack search hunts '<a command line copied from the report>'     (see: cmdstack search --help)

Install, offline use and what to export: README.md
"""


def build_review_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cmdstack review", allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"Mark groups as reviewed (routine) in a history folder's {REVIEWED_FILE}. The next run with "
                    "--history on that folder hides them. Marks stay until you remove them.",
        epilog="examples:\n"
               "  cmdstack review hunts --import Downloads\\reviewed.json\n"
               '  cmdstack review hunts --mark g1a2b3c4d5 g6f7e8d9c0 --note "IT inventory script"\n'
               "  cmdstack review hunts --remove g1a2b3c4d5\n"
               "  cmdstack review hunts --list\n"
               "Group ids (like g1a2b3c4d5) are in the group_id column of groups.csv, and on each group in the report.")
    p.add_argument("folder", metavar="FOLDER", help="the history folder used with --history")
    p.add_argument("--import", dest="imports", nargs="+", metavar="FILE", default=[],
                   help=f"{REVIEWED_FILE} files downloaded from reports, merged with the marks already saved; if a "
                        "group is already marked, the imported note replaces the saved one")
    p.add_argument("--mark", nargs="+", default=[], metavar="ID",
                   help="group ids to mark as reviewed (ids look like g1a2b3c4d5)")
    p.add_argument("--note", default="", help="why these groups are routine; applies to every id in this --mark")
    p.add_argument("--by", default=None, help=argparse.SUPPRESS)    # reviewer name; default: your user name
    p.add_argument("--remove", nargs="+", default=[], metavar="ID",
                   help="group ids to unmark; they show again from the next run")
    p.add_argument("--list", action="store_true", help="show what is marked")
    p.add_argument("--preview", action="store_true",
                   help="with --mark, --remove or --import: show what would change, without saving anything")
    return p


def _latest_titles(folder: Path) -> dict[str, str]:
    """Group id -> title from the newest run in the folder (groups.csv, else baseline.json)."""
    runs = sorted(d for d in folder.iterdir() if d.is_dir() and (d / BASELINE_FILE).exists())
    if not runs:
        return {}
    titles = {g["id"]: g.get("title", "") for g in load_baseline(runs[-1]).get("groups", [])}
    return titles


def review_main(argv) -> int:
    args = build_review_parser().parse_args(argv)
    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"cmdstack review: no folder {folder}; pass the folder you use with --history")
    target = folder / REVIEWED_FILE
    items = load_reviewed(target) if target.exists() else {}
    before = dict(items)
    for f in args.imports:
        items.update(load_reviewed(f))
    if args.mark:
        titles = _latest_titles(folder)
        unknown = [g for g in args.mark if g not in titles]
        if unknown:
            raise SystemExit(f"cmdstack review: no group {', '.join(unknown)} in the latest run in {folder}. "
                             "Use the group_id column of groups.csv (ids look like g1a2b3c4d5).")
        by = args.by or getpass.getuser()
        at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        for g in args.mark:
            items[g] = {"group_id": g, "title": titles[g], "by": by, "at": at, "note": args.note}
    for g in args.remove:
        if items.pop(g, None) is None:
            print(f"not marked: {g}", file=sys.stderr)
    if args.preview:
        for g in sorted(set(items) - set(before)):
            print(f"would mark:   {g}  {str(items[g].get('title', ''))[:80]}", file=sys.stderr)
        for g in sorted(set(before) - set(items)):
            print(f"would unmark: {g}  {str(before[g].get('title', ''))[:80]}", file=sys.stderr)
        if items == before:
            print("nothing would change", file=sys.stderr)
        print(f"preview only: {target} was not changed", file=sys.stderr)
        return 0
    if items != before:
        save_reviewed(target, items)
        added = len(set(items) - set(before))
        removed = len(set(before) - set(items))
        print(f"{target}: {len(items)} reviewed groups ({added} added, {removed} removed). "
              "The next run with --history on this folder hides them.", file=sys.stderr)
    if args.list or items == before:
        if not items:
            print("no groups are marked as reviewed", file=sys.stderr)
        for r in sorted(items.values(), key=lambda r: r.get("at", "")):
            print(f"{r['group_id']}  {r.get('at', ''):20}  {r.get('by', ''):12}  {r.get('note', '')}  |  "
                  f"{str(r.get('title', ''))[:70]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cmdstack", allow_abbrev=False, epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="cmdstack FILE [FILE ...] [options]\n"
              "       cmdstack review FOLDER [options]\n"
              '       cmdstack search FOLDER "COMMAND LINE" [options]',
        description="Group process command lines from EDR, SIEM or Sysmon exports into a short report, groups on "
                    "the fewest machines first. Your data never leaves this machine; the first run downloads the "
                    "language model (about 90 MB).\n\n"
                    "commands:\n"
                    "  cmdstack FILE ...    group an export and write the report (the options below)\n"
                    "  cmdstack review      mark groups as routine so later runs hide them (cmdstack review -h)\n"
                    "  cmdstack search      find every line like one from the report, and the machines it ran on\n"
                    "                       (cmdstack search -h)")
    p.add_argument("csv", nargs="+", metavar="FILE",
                   help="one or more exports: CSV (plain or .gz) or Excel (.xlsx), read as one run; wildcards such "
                        "as exports\\*.csv work")
    p.add_argument("-o", "--out", metavar="FOLDER",
                   help="output folder (default: cmdstack-out; with --history, a new folder such as "
                        "hunts\\2026-10-19_093000_week42)")
    p.add_argument("--full-findings", action="store_true",
                   help="put every listed group, warning-sign line and machine in findings.md (default: a "
                        "ticket-sized summary)")
    p.add_argument("--history", metavar="FOLDER",
                   help="keep runs in FOLDER: each run gets a dated subfolder, is compared with the previous run, "
                        "and hides the groups you marked with cmdstack review")
    p.add_argument("--internal-nets", default="", metavar="RANGES",
                   help="public address ranges your organization uses internally, comma separated (e.g. "
                        "142.20.0.0/16). They show as <INTERNAL_IP> instead of <PUBLIC_IP>, so internal connections "
                        "are not mistaken for internet ones. Not needed for 10.x, 172.16-31.x and 192.168.x, which are "
                        "recognized already. Use the same value every week; changing it makes groups look new")

    cols = p.add_argument_group("columns (detected from common names; name one to override it, and pass it again "
                                "every run if it is not detected)")
    cols.add_argument("--cmd-column", dest="column", metavar="NAME",
                      help="command line (e.g. CommandLine, process.command_line)")
    cols.add_argument("--host-column", dest="host", metavar="NAME",
                      help="machine name column (e.g. Computer, host, host.name). Gives each group its "
                           "machine count and names, and puts groups on the fewest machines first; without one, "
                           "groups are ranked by event count. The first lines of each run show which columns were "
                           "used. 'none' turns off a column detected wrongly")
    cols.add_argument("--count-column", dest="count", metavar="NAME",
                      help="event count per row, for exports already aggregated in the SIEM (e.g. count); "
                           "without one, each row is one event. 'none' turns off a column detected wrongly")
    # hidden: rarely needed, or only for testing; they still work (see the README)
    cols.add_argument("--hosts-count", help=argparse.SUPPRESS)       # machine count per row, without names
    cols.add_argument("--parent", help=argparse.SUPPRESS)            # parent-process column
    cols.add_argument("--label", help=argparse.SUPPRESS)             # benign/malicious, for labeled test data

    adv = p.add_argument_group("advanced")
    adv.add_argument("--baseline", help=argparse.SUPPRESS)           # --history picks these two itself
    adv.add_argument("--reviewed", help=argparse.SUPPRESS)
    adv.add_argument("--platform", choices=PLATFORMS, default="windows", help=argparse.SUPPRESS)  # linux: experimental
    adv.add_argument("--min-group", type=int, default=5, help=argparse.SUPPRESS)
    adv.add_argument("--max-shown", type=int, default=8, help=argparse.SUPPRESS)
    adv.add_argument("--model-dir", metavar="FOLDER",
                     help="folder with a downloaded copy of the language model, for machines without internet "
                          "(see Offline use in the README)")
    adv.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                     help="where to run the language model: auto (a GPU if there is one, the default), cuda (NVIDIA "
                          "GPU) or cpu")
    adv.add_argument("--threads", type=int, default=default_threads(), metavar="N",
                     help=f"CPU threads to use (default: half the cores, {default_threads()} here)")
    adv.add_argument("--match-threshold", type=float, default=MATCH_THRESHOLD, help=argparse.SUPPRESS)
    adv.add_argument("--unit", choices=("capped", "distinct", "normalized"), default="capped", help=argparse.SUPPRESS)
    adv.add_argument("--model", default=DEFAULT_MODEL, help=argparse.SUPPRESS)
    adv.add_argument("--seed", type=int, default=42, help=argparse.SUPPRESS)
    p.add_argument("--version", action="version", version=f"cmdstack {__version__}")
    return p


def expand_inputs(paths) -> list[str]:
    """Expand wildcards such as exports\\*.csv: PowerShell and Command Prompt pass them to programs unexpanded."""
    files = []
    for p in paths:
        if Path(p).exists() or not any(c in p for c in "*?["):
            files.append(p)
            continue
        matches = sorted(glob.glob(p))
        if not matches:
            raise SystemExit(f"cmdstack: no files match {p}")
        files += matches
    return list(dict.fromkeys(files))           # the same file named twice is read once


def _run_label(paths) -> str:
    """'_<export name>' for the run folder, so a folder says which export it holds, not only when it ran."""
    stem = Path(paths[0]).name
    for suffix in (".gz", ".csv", ".xlsx", ".xlsm"):
        stem = stem[: -len(suffix)] if stem.lower().endswith(suffix) else stem
    label = "".join(c if c.isalnum() or c in "-._" else "-" for c in stem).strip("-.")[:48]
    return (f"_{label}" if label else "") + (f"+{len(paths) - 1}" if len(paths) > 1 else "")


def fingerprint(path) -> dict:
    """Name, size and a hash of the first and last megabyte: enough to recognize the same export again."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with p.open("rb") as f:
        h.update(f.read(1 << 20))
        if size > 2 << 20:
            f.seek(-(1 << 20), 2)
            h.update(f.read())
    return {"name": p.name, "size": size, "sha1": h.hexdigest()[:16]}


def _same_inputs(run_dir: Path, paths) -> bool:
    try:
        old = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")).get("inputs") or []
    except (OSError, ValueError):
        return False
    return bool(old) and sorted(i["sha1"] for i in old) == sorted(fingerprint(f)["sha1"] for f in paths)


def _resolve_history(args, say):
    """With --history: pick the output folder, the previous run as baseline, and the saved reviewed list."""
    hist = Path(args.history)
    hist.mkdir(parents=True, exist_ok=True)
    runs = sorted(d for d in hist.iterdir() if d.is_dir() and (d / BASELINE_FILE).exists())
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S") + _run_label(args.csv)   # run time, then the export's name
    out = Path(args.out) if args.out else hist / stamp
    n = 1
    while not args.out and out.exists():          # never overwrite an earlier run
        n += 1
        out = hist / f"{stamp}_{n}"
    previous = [d for d in runs if d.resolve() != out.resolve()]
    if not args.baseline and previous:
        base = previous[-1]
        if _same_inputs(base, args.csv):
            # rerunning last week's export must not turn it into its own baseline (it would report nothing new)
            # marks made after that run were made on this export's lines, so they are not resurfaced by them
            try:
                args.rerun_since = json.loads((previous[-1] / BASELINE_FILE).read_text(encoding="utf-8")).get(
                    "created", "")
            except (OSError, ValueError):
                args.rerun_since = ""
            if len(previous) > 1:
                base = previous[-2]
                say(f"this is the same export as the last run ({previous[-1].name}); comparing with the run "
                    f"before it instead: {base}")
            else:
                base = None
                say(f"this is the same export as the only earlier run ({previous[-1].name}), so there is nothing "
                    "earlier to compare with: everything is new")
        if base is not None:
            args.baseline = str(base)
            if base == previous[-1]:
                say(f"comparing with the previous run: {base}")
    elif not previous:
        say("first run in this history folder: everything is new; later runs will be compared with this one")
    if not args.reviewed and (hist / REVIEWED_FILE).exists():
        args.reviewed = str(hist / REVIEWED_FILE)
        say(f"applying the reviewed list: {hist / REVIEWED_FILE}")
    return out, hist / ".cache"


def _warn_volume(base_dir: Path, events: int, say) -> None:
    """Say so when this export is much smaller or larger than the baseline's, which usually means it covers a
    different time span (three days against a week): fewer events means fewer known groups show up, more means
    more lines look new simply because there is more data."""
    try:
        old = int(json.loads((base_dir / "summary.json").read_text(encoding="utf-8"))["events"])
    except (OSError, ValueError, KeyError):
        return
    if old and not 0.5 <= events / old <= 2:
        say(f"note: this export has {events:,} events against {old:,} in the baseline ({events / old:.0%}). If it "
            "covers a shorter or longer time span, expect fewer known groups or more new lines for that reason alone")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "review" and not Path("review").is_file():     # a file named "review" is still an input
        return review_main(argv[1:])
    if argv and argv[0] == "search" and not Path("search").is_file():
        return search_main(argv[1:])
    args = build_parser().parse_args(argv)
    args.csv = expand_inputs(args.csv)
    start = time.time()
    say = lambda msg: print(f"[{time.time() - start:6.1f}s] {msg}", file=sys.stderr)
    if args.platform == "linux":
        say("note: linux normalization is experimental")

    if args.history:
        out, cache = _resolve_history(args, say)
    else:
        out = Path(args.out or "cmdstack-out")
        cache = out / ".cache"

    lines, info = read_events(args.csv, column=args.column, count=args.count, host=args.host,
                              label=args.label, parent=args.parent, hosts_count=args.hosts_count)
    info["inputs"] = [fingerprint(f) for f in args.csv]
    baseline = load_baseline(args.baseline) if args.baseline else None
    if args.baseline:
        _warn_volume(Path(args.baseline), int(lines["events"].sum()), say)
    reviewed = load_reviewed(args.reviewed) if args.reviewed else None
    since = getattr(args, "rerun_since", "")
    covered = {g for g, r in (reviewed or {}).items() if since and r.get("at", "") >= since}
    if covered:
        say(f"{len(covered):,} groups were marked reviewed after the last run of this export, so they stay hidden "
            "even where their lines are new since the baseline")
    used =[f"command line {info['command_line_column']!r}"]
    for name, key in (("host", "host_column"), ("count", "count_column"), ("hosts-count", "hosts_count_column"),
                      ("parent", "parent_column"), ("label", "label_column")):
        if info.get(key):
            used.append(f"{name} {info[key]!r}" + (" (detected)" if name in info["detected"] else ""))
    say(f"read {info['rows_read']:,} rows -> {len(lines):,} distinct command lines; columns: {', '.join(used)}")
    if info.get("lenient"):
        say("WARNING: the export is not clean CSV (for example rows with extra fields, or files with different line "
            "endings pasted together), so it was read leniently: a malformed row is kept with its extra fields "
            "dropped instead of being skipped. Spot-check odd command lines in the report")
    if info.get("rows_skipped"):
        say(f"WARNING: {info['rows_skipped']:,} rows could not be parsed (for example more fields than the header) "
            "and were skipped; they are not in the report. Check the export for unquoted commas or line breaks")
    if not info.get("host_column") and not info.get("hosts_count_column"):
        say("no host column found, so groups are ranked by event count; name one with --host-column to put groups "
            "on the fewest machines first")
    elif not info.get("host_column"):
        say('the export has machine counts but no names: group counts are a lower bound ("at least N") and '
            "machines are not listed; export one row per command line and machine for exact counts and names")

    with limit_threads(args.threads):
        encode = load_encoder(args.model, args.model_dir, args.device, log=say)
    result = run(lines, encode, platform=args.platform, internal_nets=parse_networks(args.internal_nets),
                 unit=args.unit, min_group=args.min_group, max_shown=args.max_shown, seed=args.seed,
                 cache_dir=cache, model=args.model, threads=args.threads, log=say)
    settings = {k: v for k, v in vars(args).items() if k not in ("csv", "out")}
    summary = write_outputs(result, info, out, settings, baseline=baseline, reviewed=reviewed,
                            threshold=args.match_threshold, covered=covered)
    say(f"{summary['normalized_lines']:,} normalized lines -> {summary['groups']:,} groups, "
        f"{summary['noise_lines']:,} noise lines, {summary['representative_lines']:,} representative lines")
    if summary.get("status") and baseline is not None:
        st = summary["status"]
        say(f"compared with the baseline: {st['new']:,} new, {st['changed']:,} changed, {st['known']:,} known, "
            f"{st['reviewed']:,} reviewed groups; {summary['new_noise_lines']:,} noise lines not seen before")
        if st.get(REVIEWED_NEW):
            say(f"{st[REVIEWED_NEW]:,} reviewed groups have lines not seen before; they are shown again")
    print(f"\nOpen the report: {(out / 'report.html').resolve()}", file=sys.stderr)
    if args.history:
        print(f'To hide routine groups next time: cmdstack review "{args.history}" --mark <id> --note "why", '
              f"or mark them in the report and run cmdstack review \"{args.history}\" --import <downloaded "
              f"{REVIEWED_FILE}>", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
