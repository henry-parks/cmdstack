"""Write the outputs: lines.csv.gz, groups.csv, noise.csv, representatives.csv and a self-contained report.html."""
from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import baseline as bl
from .io import HOST_SAMPLE
from .normalize import program
from .search import write_index

MIXED_SEPARATOR = "|"
RAW_SHOWN = 3            # raw command lines shown under each representative line in the report
RAW_CHARS = 2_000        # longer raw lines are cut in the report (lines.csv.gz keeps them whole)
MINORITY_SHOWN = 20     # lines listed per mixed group under "Lines labeled ..."
PROGRAM_CHIPS = 20      # programs offered as filter chips above the groups
METHOD_NAMES = {"random": "typical", "farthest_first": "unusual", "mmr": "varied"}   # plain names in the report
LOOK_SHOWN = 12          # lines listed per group under "Lines worth a look"
OTHER_PROGRAMS_SHOWN = 4  # programs named next to a group's title besides its main one

# a script or program launched from a folder any user can write to (temp, Downloads, Desktop, Public, Windows\Temp,
# PerfLogs, the recycle bin; /tmp and /dev/shm on Linux). Routine installers do this too: it is a reading cue,
# not a verdict. ProgramData and AppData\Roaming are left out because many legitimate agents live there.
USER_WRITABLE_RE = re.compile(
    r"(?i)(?:\\users\\[^\\\"]+\\(?:appdata\\local\\temp|downloads|desktop)|\\windows\\temp|\\users\\public"
    r"|\\perflogs|\\\$recycle\.bin"
    r"|%(?:temp|tmp|public)%|\$env:(?:temp|tmp|public)|(?:%userprofile%|\$env:userprofile|\$home|~)\\(?:downloads|desktop))"
    r"\\[^\"|&<>]*?\.(?:exe|com|bat|cmd|ps1|vbs|vbe|js|jse|wsf|hta|scr|dll)(?!\w)"
    r"|(?:^|[\s\"'])/(?:tmp|var/tmp|dev/shm)/\S+")

# Base64 alone is not a warning sign: many programs pass base64 arguments (one developer workstation had 12.5% of
# its lines decode to text). An encoded line counts when PowerShell is handed it with -enc (or a prefix of
# -EncodedCommand), or when the decoded text reads like a command or script.
ENCODED_SWITCH_RE = re.compile(r"(?i)(?:powershell|pwsh)\S*\s(?:.*?\s)?[-/]e(?:c|nc?\w*)?\s+[\"']?[a-z0-9+/=]{16}")
COMMAND_LIKE_RE = re.compile(
    r"(?i)\b(?:iex|(?:get|set|new|add|remove|start|stop|invoke|out|write|copy|move)-[a-z]{3,}|frombase64string|downloadstring|downloadfile"
    r"|net\.webclient|system\.(?:net|reflection|convert)|set-executionpolicy|-(?:nop|noprofile|windowstyle|exec))\b"
    r"|\$\w+\s*=|\bcmd(?:\.exe)?\s+/[ck]\b|\b(?:powershell|pwsh|rundll32|regsvr32|mshta|certutil)(?:\.exe)?\s")


def encoded_command(command_line: str, decoded: str) -> bool:
    """True for an encoded line worth a look: decodable base64 given to PowerShell -enc, or decoding to a command."""
    return bool(decoded) and bool(ENCODED_SWITCH_RE.search(command_line) or COMMAND_LIKE_RE.search(decoded))


def _label_set(values) -> list[str]:
    return sorted({v for s in values.dropna() for v in str(s).split(MIXED_SEPARATOR) if v})


def _groups_table(lines: pd.DataFrame, reps: dict, has_labels: bool, has_hosts: bool) -> pd.DataFrame:
    grouped = lines[lines["group"] >= 0].groupby("group")
    total_events = max(int(lines["events"].sum()), 1)
    rows = []
    for gid, g in grouped:
        top_template = g.groupby("template")["events"].sum().sort_values(ascending=False)
        programs = g.groupby("program")["events"].sum().sort_values(ascending=False).head(3)
        top_line = g.groupby("normalized")["events"].sum().sort_values(ascending=False)
        # the title is the most frequent line, so name the other programs in the group next to it: a small
        # group of dropped executables can share a group (and a title) with a common system process
        all_programs = g.groupby("program")["events"].sum().sort_values(ascending=False)
        title_program = g.loc[g["normalized"] == top_line.index[0], "program"].iloc[0]
        others = [p for p in all_programs.index if p != title_program]
        row = {
            "group": int(gid),
            "title": top_line.index[0],             # the group's most frequent normalized line, real digits kept
            "other_programs": ", ".join(others),
            "user_writable_lines": int(g["user_writable"].sum()),
            "encoded_lines": int(g["encoded"].sum()),
            "distinct_lines": int(len(g)),
            "normalized_lines": int(g["normalized"].nunique()),
            "events": int(g["events"].sum()),
            "share_of_events": round(100 * g["events"].sum() / total_events, 3),
            "main_template": top_template.index[0],
            "main_template_share": round(100 * top_template.iloc[0] / max(top_template.sum(), 1), 1),
            "top_programs": ", ".join(programs.index),
            "representatives": " || ".join(text for _, text in reps.get(int(gid), [])),
            "obfuscated_lines": int(((g["caret_count"] > 0) | g["non_ascii"] | (g["b64_decoded"] != "")).sum()),
        }
        if has_hosts:
            row["max_hosts_per_line"] = int(g["hosts"].max())
            row["hosts"], row["hosts_note"] = _group_hosts(g)
            row["host_names"] = ", ".join(_host_names(g))        # every machine, for scoping (capped per line)
        if has_labels:
            values = _label_set(g["labels"])
            row["labels"] = MIXED_SEPARATOR.join(values)
            row["mixed"] = len(values) > 1
            if row["mixed"]:
                # the label most lines carry; the others are highlighted in the report
                counts = g["labels"].dropna().str.split(MIXED_SEPARATOR).explode().value_counts()
                row["majority_label"] = counts.index[0]
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # two reading orders, mixed-label groups first in both: by volume (most events), and by rarity (fewest hosts,
    # then fewest events), which brings activity seen on one machine a handful of times to the top
    tier = table["mixed"].map({True: 0, False: 1}) if has_labels else 0
    spread = table["hosts"] if has_hosts else 0
    table["volume_rank"] = (pd.DataFrame({"t": tier, "e": -table["events"]})
                            .sort_values(["t", "e"], kind="stable").index.argsort() + 1)
    table["rarity_rank"] = (pd.DataFrame({"t": tier, "h": spread, "e": table["events"]})
                            .sort_values(["t", "h", "e"], kind="stable").index.argsort() + 1)
    order = "rarity_rank" if has_hosts else "volume_rank"
    return table.sort_values(order, kind="stable").reset_index(drop=True)


FEW_HOSTS = 3   # name the machines when something ran on this many or fewer


def _host_names(g: pd.DataFrame) -> list[str]:
    """Every machine name seen for a group's lines (each line keeps up to HOST_SAMPLE names)."""
    if "host_sample" not in g.columns:
        return []
    return sorted({str(n) for sample in g["host_sample"] if sample is not None for n in sample})


def _sample_text(sample) -> str:
    return "; ".join(sorted(str(n) for n in sample)) if sample is not None else ""


def _group_hosts(g: pd.DataFrame) -> tuple[int, str]:
    """A group's host count and how to read it: "" exact, "+" capped at HOST_SAMPLE, "at least" a lower bound.

    With host names, distinct hosts are counted across the group's lines (up to HOST_SAMPLE). With only a host
    count per line (an aggregated export), the widest line is a lower bound for the group.
    """
    widest = int(g["hosts"].max())
    if "host_sample" not in g.columns:
        return widest, "at least"
    if widest >= HOST_SAMPLE:
        return HOST_SAMPLE, "+"
    names = set()
    for sample in g["host_sample"]:
        if sample is not None:
            names.update(sample)
            if len(names) >= HOST_SAMPLE:
                return HOST_SAMPLE, "+"
    return max(len(names), widest), ""


def _rep_details(lines: pd.DataFrame, reps: dict, has_labels: bool, has_hosts: bool):
    """Stats per representative line, and the most frequent raw lines behind each one."""
    shown = {text for items in reps.values() for _, text in items}
    sub = lines[lines["normalized"].isin(shown)]
    agg = {"events": ("events", "sum"), "raw_lines": ("command_line", "size")}
    if has_hosts:
        agg["max_hosts"] = ("hosts", "max")
    stats = sub.groupby("normalized").agg(**agg)
    if has_labels:
        stats["labels"] = sub.groupby("normalized")["labels"].agg(lambda s: MIXED_SEPARATOR.join(_label_set(s)))
    raw_cols = ["normalized", "command_line", "events"] + (["hosts"] if has_hosts else [])
    top = sub.sort_values("events", ascending=False, kind="stable").groupby("normalized").head(RAW_SHOWN)[raw_cols]
    raw = {t: g.drop(columns="normalized").to_dict("records") for t, g in top.groupby("normalized", sort=False)}
    return stats, raw


def _minority_lines(lines: pd.DataFrame, groups: pd.DataFrame) -> dict:
    """For each mixed group: (other labels, normalized lines carrying them, most events first).

    Representative lines are picked without labels, so a few malicious lines in a mostly benign group can be
    missing from them; this lists those lines directly.
    """
    if groups.empty or "majority_label" not in groups.columns:
        return {}
    out = {}
    for _, row in groups[groups["mixed"]].iterrows():
        majority = row["majority_label"]
        g = lines[lines["group"] == row["group"]]
        has_other = g["labels"].fillna("").map(lambda s: any(v and v != majority for v in s.split(MIXED_SEPARATOR)))
        sub = g[has_other]
        table = (sub.groupby("normalized")
                 .agg(events=("events", "sum"), raw_lines=("command_line", "size"))
                 .sort_values("events", ascending=False, kind="stable").reset_index())
        others = [v for v in _label_set(sub["labels"]) if v != majority]
        out[int(row["group"])] = (others, table)
    return out


def _group_hosts_table(lines: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """One row per group and machine: how many of the group's distinct lines ran there (for scoping)."""
    sub = lines.loc[lines["group"] >= 0, ["group", "host_sample"]].explode("host_sample").dropna()
    table = sub.groupby(["group", "host_sample"]).size().rename("lines").reset_index().rename(
        columns={"host_sample": "host"})
    cols = ["group", "group_id", "title"] + (["status"] if "status" in groups.columns else [])
    return (groups[cols].merge(table, on="group")
            .sort_values(["group", "lines", "host"], ascending=[True, False, True], kind="stable"))


def _look_lines(lines: pd.DataFrame, groups: pd.DataFrame, seen_before: pd.Series | None) -> dict:
    """Per group, the lines an analyst should not miss behind its title: lines run from user-writable folders,
    encoded lines, lines from programs other than the title's, and (against a baseline) lines not seen before in a
    group that was. Returns {group: (rows, total)}; rows are the first LOOK_SHOWN, most telling first."""
    if groups.empty:
        return {}
    status = dict(zip(groups["group"], groups["status"])) if "status" in groups.columns else {}
    others = {int(r["group"]): {p for p in str(r.get("other_programs") or "").split(", ") if p}
              for _, r in groups.iterrows()}
    grouped = lines[lines["group"] >= 0]
    # one boolean column per reason, most telling first; the weights rank lines with more and stronger reasons
    flags = pd.DataFrame({
        "runs from a temp or user folder": grouped["user_writable"],
        "encoded": grouped["encoded"],
        "not seen before": (~seen_before[grouped.index] & grouped["group"].map(
            lambda gid: status.get(gid) not in ("", "new", None))) if seen_before is not None else False,
        "other program": pd.Series([p in others.get(gid, ()) for gid, p in zip(grouped["group"], grouped["program"])],
                                   index=grouped.index),
    })
    weights = pd.Series([8, 4, 2, 1], index=flags.columns)
    score = flags.astype(int).mul(weights).sum(axis=1)
    flagged = grouped[score > 0].assign(_score=score[score > 0])
    out = {}
    for gid, g in flagged.groupby("group"):
        sort = ["_score", "hosts", "events"] if "hosts" in g.columns else ["_score", "events"]
        top = g.sort_values(sort, ascending=[False] + [True] * (len(sort) - 1), kind="stable").head(LOOK_SHOWN)
        why = flags.loc[top.index].apply(lambda r: ", ".join(c for c in flags.columns if r[c]), axis=1)
        out[int(gid)] = (top.assign(reasons=why), len(g))
    return out


# findings.md sizes: (short, full). The short form is meant to be pasted into a ticket as it is; --full-findings
# keeps every new group, every flagged line shown in the report and the full host lists.
FINDINGS_GROUPS = (5, 10_000)    # groups written out in full (with a baseline: new, changed, reviewed with new lines)
FINDINGS_BRIEF = (10, 0)         # further groups listed one line each
FINDINGS_LOOK = (3, LOOK_SHOWN)  # flagged lines per group
FINDINGS_HOSTS = (8, 10_000)     # machine names per group (all are in group_hosts.csv)
FINDINGS_NOISE = (8, 30)         # one-off lines
FINDINGS_CHARS = (160, 400)      # characters of a command line


def _n(n, word: str) -> str:
    return f"{int(n):,} {word}{'' if int(n) == 1 else 's'}"


def _brief_signs(row: pd.Series) -> str:
    """'; warning signs: ...' for a one-line group, so an attack inside a big routine group is not a bare title."""
    signs = []
    row = row.where(pd.notna(row), 0)
    if int(row.get("user_writable_lines") or 0):
        signs.append(_n(row["user_writable_lines"], "line") + " from temp or user folders")
    if int(row.get("encoded_lines") or 0):
        signs.append(_n(row["encoded_lines"], "encoded line"))
    text = "; warning signs: " + ", ".join(signs) if signs else ""
    others = [p for p in str(row.get("other_programs") or "").split(", ") if p]
    if others:                                   # common in routine groups too, so listed but not called a warning
        text += "; other programs: " + ", ".join(others[:OTHER_PROGRAMS_SHOWN]) + (
            f" and {len(others) - OTHER_PROGRAMS_SHOWN} more" if len(others) > OTHER_PROGRAMS_SHOWN else "")
    return text


def _md_code(text: str, limit: int = 300) -> str:
    """Inline code that survives backticks inside the command line (PowerShell's escape character)."""
    text = " ".join(str(text).split())
    text = text if len(text) <= limit else text[:limit] + " ..."
    return f"`` {text} ``"


def _findings_md(summary: dict, groups: pd.DataFrame, noise: pd.DataFrame, look: dict, lines: pd.DataFrame,
                 has_baseline: bool, has_hosts: bool, full: bool = False) -> str:
    """A Markdown summary to paste into a ticket or case notes: what to read, where it ran, the lines that matter.

    Groups with warning signs (lines from temp or user folders, encoded lines, other programs) come first.
    """
    k = 1 if full else 0
    out = ["# cmdstack findings", "",
           f"Generated {summary['generated']} from {', '.join(summary['files'])}: {summary['events']:,} events, "
           f"{summary['distinct_lines']:,} distinct command lines, {summary['groups']:,} groups, "
           f"{summary['noise_lines']:,} ungrouped lines."]
    if has_baseline and summary.get("status"):
        st = summary["status"]
        out.append(f"Compared with the previous run: {st['new']:,} new groups, {st['changed']:,} changed, "
                   f"{st[bl.REVIEWED_NEW]:,} reviewed groups with new lines, {summary['new_noise_lines']:,} "
                   "ungrouped lines not seen before.")
        pool = groups[groups["status"].isin(["new", "changed", bl.REVIEWED_NEW])] if not groups.empty else groups
        heading = "Groups that are new or changed since the previous run"
    else:
        pool = groups
        heading = "Rarest groups" if has_hosts else "Largest groups"
    if not pool.empty:
        warned = pool["group"].map(lambda g: int(g) in look and look[int(g)][1] > 0
                                   and any(r != "not seen before" for r in look[int(g)][0]["reasons"]))
        pool = pool.assign(_w=~warned).sort_values(["_w"], kind="stable").drop(columns="_w")
    pick = pool.head(FINDINGS_GROUPS[k])
    brief = pool.iloc[len(pick):len(pick) + FINDINGS_BRIEF[k]]
    more_groups = len(pool) - len(pick) - len(brief)
    code = lambda text: _md_code(text, FINDINGS_CHARS[k])
    shown = len(pick) + len(brief)
    out += ["", f"## {heading} ({len(pool):,}{f', {shown} here' if shown < len(pool) else ''})", ""]
    if pick.empty:
        out.append("None.")
    for _, row in pick.iterrows():
        gid = int(row["group"])
        status = f", {row['status']}" if row.get("status") else ""
        out.append(f"### Group {gid} ({row['group_id']}{status})")
        out.append(f"- Main line: {code(row['title'])}")
        programs = list(dict.fromkeys([program(str(row["title"]))] + str(row["top_programs"]).split(", ")
                                      + [p for p in str(row.get("other_programs") or "").split(", ") if p]))
        out.append(f"- Programs: {', '.join(programs)}")
        bits = [_n(row["events"], "event"), _n(row["distinct_lines"], "distinct line")]
        if has_hosts:
            names = [n for n in str(row.get("host_names") or "").split(", ") if n]
            cap = FINDINGS_HOSTS[k]
            shown = ", ".join(names[:cap]) + (f" and {len(names) - cap:,} more (group_hosts.csv)"
                                              if len(names) > cap else "")
            approx = "at least " if row.get("hosts_note") == "at least" else ""
            host_count = (f"{int(row['hosts']):,}{'+' if row.get('hosts_note') == '+' else ''} "
                          f"host{'s' if int(row['hosts']) != 1 else ''}")
            bits.append(approx + host_count + (f": {shown}" if shown else ""))
        out.append("- " + "; ".join(bits))
        if row.get("status") == "changed" and pd.notna(row.get("hosts_before")):
            out.append(f"- Spread: {int(row['hosts_before']):,} hosts in the previous run, {int(row['hosts']):,} now")
        if gid in look:
            rows, total = look[gid]
            shown_rows = rows.head(FINDINGS_LOOK[k])
            out.append(f"- Lines worth a look ({total:,}" + (f", {len(shown_rows)} shown" if total > len(shown_rows)
                                                              else "") + "):")
            for _, r in shown_rows.iterrows():
                where = ""
                if has_hosts and r.get("host_names") and 0 < int(r["hosts"]) <= FEW_HOSTS:
                    where = f" on {str(r['host_names']).replace('; ', ', ')}"
                out.append(f"  - {code(r['command_line'])} ({r['reasons']}; {_n(r['events'], 'event')}{where})")
                if r["b64_decoded"]:
                    out.append(f"    - decoded: {code(r['b64_decoded'])}")
        elif full or gid not in look:
            examples = lines[lines["group"] == gid].sort_values("events", ascending=False).head(3 if full else 1)
            label = "Most frequent lines" if full else "Most frequent line"
            out.append(f"- {label}:")
            for _, r in examples.iterrows():
                out.append(f"  - {code(r['command_line'])} ({_n(r['events'], 'event')})")
        out.append("")
    if len(brief):
        out += ["Also new or changed:" if has_baseline else "Next:", ""]
        for _, row in brief.iterrows():
            hosts = f", {_n(row['hosts'], 'host')}" if has_hosts else ""
            out.append(f"- Group {int(row['group'])} ({row['group_id']}): {code(row['title'])}{hosts}"
                       + _brief_signs(row))
        out.append("")
    if more_groups > 0:
        out += [f"{more_groups:,} more groups are in the report and groups.csv (or rerun with --full-findings).", ""]
    if has_baseline and "seen_before" in noise.columns:
        rest = noise[~noise["seen_before"]]
        heading = "Ungrouped lines not seen before"
    elif has_hosts:
        rest = noise[noise["hosts"] <= FEW_HOSTS]
        heading = f"Ungrouped lines on {FEW_HOSTS} hosts or fewer"
    else:
        rest = noise
        heading = "Ungrouped lines"
    out += [f"## {heading} ({len(rest):,})", ""]
    for _, r in rest.head(FINDINGS_NOISE[k]).iterrows():
        where = ""
        if has_hosts and r.get("host_names") and 0 < int(r["hosts"]) <= FEW_HOSTS:
            where = f", {str(r['host_names']).replace('; ', ', ')}"
        out.append(f"- {code(r['command_line'])} ({_n(r['events'], 'event')}{where})")
    if len(rest) > FINDINGS_NOISE[k]:
        out.append(f"- and {len(rest) - FINDINGS_NOISE[k]:,} more in noise.csv")
    out += ["", "cmdstack groups command lines; it does not decide what is malicious. Confirm with the raw telemetry."]
    return "\n".join(out) + "\n"


def write_outputs(result: dict, info: dict, out_dir: Path, settings: dict, baseline: dict | None = None,
                  reviewed: dict | None = None, threshold: float = bl.MATCH_THRESHOLD,
                  covered: set | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: pd.DataFrame = result["lines"]
    reps: dict = result["representatives"]
    has_labels = "labels" in lines.columns
    has_hosts = "hosts" in lines.columns

    lines = lines.assign(user_writable=lines["command_line"].astype(str).str.contains(USER_WRITABLE_RE),
                         encoded=[encoded_command(str(c), d) for c, d in zip(lines["command_line"], lines["b64_decoded"])])
    if "host_sample" in lines.columns:
        lines = lines.assign(host_names=lines["host_sample"].map(_sample_text))
    keep = ["command_line", "events", "group", "normalized", "template", "program", "user_writable", "encoded",
            "b64_decoded", "caret_count", "whitespace_runs", "non_ascii", "unicode_changed"]
    keep += [c for c in ("hosts", "host_names", "labels", "parent") if c in lines.columns]
    lines_out = lines[keep].sort_values(["group", "events"], ascending=[True, False])
    # every distinct raw line can run to millions of rows, so this one is compressed (pandas reads it directly)
    lines_out.to_csv(out_dir / "lines.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 3})
    stale = out_dir / "lines.csv"            # left by an older cmdstack version; would no longer match this run
    if stale.exists():
        stale.unlink()

    groups = _groups_table(lines, reps, has_labels, has_hosts)
    has_status = baseline is not None or bool(reviewed)
    groups, new_baseline, seen_before = bl.compare(groups, result["centroids"], lines, reps, result["model"],
                                                   baseline, reviewed, threshold, covered)
    bl.write_baseline(new_baseline, out_dir)
    stale = out_dir / "allowlist.csv"            # written by earlier versions; no longer produced
    if stale.exists():
        stale.unlink()
    groups.to_csv(out_dir / "groups.csv", index=False)
    if "vectors" in result:
        write_index(out_dir, result["texts"], result["vectors"])
    if "host_sample" in lines.columns and not groups.empty:
        _group_hosts_table(lines, groups).to_csv(out_dir / "group_hosts.csv", index=False)
    look = _look_lines(lines, groups, seen_before if baseline is not None else None)

    noise = lines[lines["group"] < 0][[c for c in keep if c != "group"]]
    if baseline is not None:
        # lines not seen in the baseline first: one-off commands that are new are what this list is for
        noise = noise.assign(seen_before=seen_before[noise.index]).sort_values(["seen_before", "events"],
                                                                               ascending=[True, False])
    else:
        noise = noise.sort_values("events", ascending=False)
    noise.to_csv(out_dir / "noise.csv", index=False)

    stats, raw = _rep_details(lines, reps, has_labels, has_hosts)
    rep_rows = []
    for gid, items in reps.items():
        for i, (method, text) in enumerate(items):
            row = {"group": gid, "rank": i + 1, "method": method, "normalized": text}
            row.update(stats.loc[text].to_dict())
            rep_rows.append(row)
    pd.DataFrame(rep_rows).to_csv(out_dir / "representatives.csv", index=False)

    summary = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "files": info["files"],
        "inputs": info.get("inputs", []),
        "rows_read": info["rows_read"],
        "rows_skipped": info.get("rows_skipped", 0),
        "read_leniently": bool(info.get("lenient")),
        "events": int(lines["events"].sum()),
        "distinct_lines": int(len(lines)),
        "normalized_lines": int(lines["normalized"].nunique()),
        "templates": int(lines["template"].nunique()),
        "groups": int(len(groups)),
        "noise_lines": int(len(noise)),
        "noise_events": int(noise["events"].sum()),
        "representative_lines": int(sum(len(v) for v in reps.values())),
        "mixed_groups": int(groups["mixed"].sum()) if has_labels and not groups.empty else None,
        "status": ({k: int((groups["status"] == k).sum()) for k in bl.STATUSES}
                   if has_status and not groups.empty else None),
        "new_noise_lines": int((~noise["seen_before"]).sum()) if "seen_before" in noise.columns else None,
        "settings": settings,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    minority = _minority_lines(lines, groups) if has_labels else {}
    prior = {**{g["id"]: g["reviewed"] for g in (baseline or {}).get("groups", []) if g.get("reviewed")},
             **(reviewed or {})}
    page = _html(summary, groups, noise, reps, stats, raw, minority, has_labels, has_hosts,
                 has_status, baseline is not None, prior, look)
    (out_dir / "report.html").write_text(page, encoding="utf-8")
    (out_dir / "findings.md").write_text(
        _findings_md(summary, groups, noise, look, lines, baseline is not None, has_hosts,
                     full=bool(settings.get("full_findings"))), encoding="utf-8")
    return summary


_CSS = """
:root{--bg:#fbfbfa;--fg:#16191d;--muted:#5d6570;--line:#d5d8dd;--card:#f2f3f4;--warn:#9a5407;--accent:#2457a6}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--fg:#e6e8eb;--muted:#9aa1ab;--line:#2c3138;--card:#1b1e22;--warn:#e0954a;--accent:#7fa8e8}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,Arial,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:18px;margin:28px 0 8px}
.meta,.sub{color:var(--muted);font-size:13px} .cards{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}
.card{background:var(--card);padding:10px 14px;border-radius:6px;min-width:120px}
.card b{display:block;font-size:20px} .card span{color:var(--muted);font-size:12px}
.tools{position:sticky;top:0;z-index:1;background:var(--bg);display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:8px 0;border-bottom:1px solid var(--line)}
.tools select{font:inherit;font-size:13px;padding:3px 4px;border:1px solid var(--line);border-radius:4px;background:var(--card);color:var(--fg)}
.tools input[type=search]{flex:1 1 260px;font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:5px;background:var(--card);color:var(--fg)}
button{font:12px system-ui,Segoe UI,Arial,sans-serif;padding:3px 9px;border:1px solid var(--line);border-radius:4px;background:var(--card);color:var(--fg);cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.grp{display:flex;gap:8px;align-items:flex-start;border-top:1px solid var(--line);padding:8px 0} .grp>input{margin-top:5px}
.grp>details{flex:1;min-width:0} summary{cursor:pointer;display:flex;flex-wrap:wrap;gap:8px;align-items:baseline}
summary .meta{margin-left:auto} .gid{color:var(--muted);font-size:13px}
code{font:12.5px/1.4 Consolas,ui-monospace,monospace;overflow-wrap:anywhere}
ol{margin:6px 0 4px 18px;padding:0} ol>li{margin:8px 0}
.line{display:flex;gap:8px;align-items:flex-start} .line code{flex:1;min-width:0}
.cp{padding:0 6px;font-size:11px;color:var(--muted)}
.stats{color:var(--muted);font-size:12px}
.raw summary{display:list-item;font-size:12px;color:var(--accent)}
.raw table{margin:4px 0 2px}
.raw code,#noise code{display:block;max-height:7.5em;overflow:auto}
.mixed,.hot{color:var(--warn);font-weight:600}
.mixed{font-size:12px;border:1px solid var(--warn);border-radius:4px;padding:0 5px}
table{border-collapse:collapse;width:100%} td{border-top:1px solid var(--line);padding:4px 6px;vertical-align:top}
td.n{text-align:right;white-space:nowrap;color:var(--muted);font-size:12px}
.chips{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:8px 0}
.chip{border-radius:12px;padding:2px 10px} .chip span{color:var(--muted)}
.chip.on{border-color:var(--accent);color:var(--accent);font-weight:600}
.minority{border-left:3px solid var(--warn);padding:2px 0 2px 10px;margin:6px 0 10px}
.st{font-size:11px;font-weight:600;border-radius:4px;padding:0 5px;border:1px solid var(--line);color:var(--muted)}
.st-new{color:var(--accent);border-color:var(--accent)} .st-changed,.st-renew{color:var(--warn);border-color:var(--warn)}
.flag{font-size:11px;font-weight:600;color:var(--warn)} .also{color:var(--muted);font-size:12px}
.look{border-left:3px solid var(--warn);padding:2px 0 2px 10px;margin:6px 0 10px}
.hosts summary{display:list-item;font-size:12px;color:var(--accent)} .hosts p{margin:4px 0;font-size:12px}
.guide{margin:4px 0 10px;font-size:13px;color:var(--muted)} .guide summary{display:list-item;color:var(--accent)}
.guide ul{margin:6px 0 0 18px;padding:0} .guide li{margin:3px 0}
.hide{display:none!important}
"""

_JS = """
const groups=[...document.querySelectorAll('.grp')], noiseRows=[...document.querySelectorAll('#noise tr')];
const hay=new Map([...groups,...noiseRows].map(el=>[el,el.textContent.toLowerCase()]));
const q=document.getElementById('q'), mixedOnly=document.getElementById('mixedonly'), shownCount=document.getElementById('shown');
const exportBtn=document.getElementById('export'), chips=[...document.querySelectorAll('.chip')];
let prog=''; const showSel=document.getElementById('show'), markBtn=document.getElementById('markrev');
function applyFilter(){
  const term=q.value.trim().toLowerCase(); let n=0;
  const show=showSel?showSel.value:'all';
  for(const g of groups){const st=g.dataset.status;
    const okShow=show==='all'||(show==='nr'&&st!=='reviewed')||(show==='nc'&&(st==='new'||st==='changed'||st==='reviewed, new lines'));
    const ok=okShow&&(!term||hay.get(g).includes(term))&&(!mixedOnly||!mixedOnly.checked||g.dataset.mixed)&&(!prog||g.dataset.p===prog);
    g.classList.toggle('hide',!ok); if(ok)n++;}
  for(const r of noiseRows){r.classList.toggle('hide',!!term&&!hay.get(r).includes(term));}
  shownCount.textContent=n.toLocaleString()+' of '+groups.length.toLocaleString()+' groups shown';
}
function selected(){return groups.filter(g=>g.querySelector(':scope>input').checked);}
function updateExport(){const n=selected().length;exportBtn.disabled=!n;exportBtn.textContent='Export selected ('+n+')';
  markBtn.disabled=!n;markBtn.textContent='Mark reviewed ('+n+')';}
chips.forEach(c=>c.addEventListener('click',()=>{prog=(prog===c.dataset.p)?'':c.dataset.p;
  chips.forEach(x=>x.classList.toggle('on',x.dataset.p===prog));applyFilter();}));
q.addEventListener('input',applyFilter); if(showSel)showSel.addEventListener('change',applyFilter);
markBtn.addEventListener('click',()=>{
  const by=prompt('Reviewed by (name or initials):',''); if(by===null)return;
  const note=prompt('Note for the selected groups (optional):','')||'';
  const prior=JSON.parse(document.getElementById('prior-reviewed').textContent).reviewed||[];
  const at=new Date().toISOString().slice(0,16).replace('T',' ')+' UTC', ids=new Set();
  const items=selected().map(g=>{ids.add(g.dataset.id);return {group_id:g.dataset.id,title:g.querySelector('summary code').textContent,by:by,at:at,note:note};});
  const out={reviewed:prior.filter(r=>!ids.has(r.group_id)).concat(items)};
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));
  a.download='reviewed.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
}); if(mixedOnly)mixedOnly.addEventListener('change',applyFilter);
document.getElementById('selshown').addEventListener('click',()=>{
  const vis=groups.filter(g=>!g.classList.contains('hide')), on=!vis.every(g=>g.querySelector(':scope>input').checked);
  vis.forEach(g=>g.querySelector(':scope>input').checked=on); updateExport();});
document.addEventListener('change',e=>{if(e.target.matches('.grp>input'))updateExport();});
function copyText(t){
  if(navigator.clipboard&&window.isSecureContext)return navigator.clipboard.writeText(t);
  const a=document.createElement('textarea');a.value=t;document.body.appendChild(a);a.select();
  document.execCommand('copy');a.remove();return Promise.resolve();
}
document.addEventListener('click',e=>{
  const b=e.target.closest('.cp'); if(!b)return; e.preventDefault();
  copyText(b.parentElement.querySelector('code').textContent).then(()=>{b.textContent='copied';setTimeout(()=>b.textContent='copy',1200);});
});
const csv=v=>{v=String(v??'');return /[",\\n\\r]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v;};
exportBtn.addEventListener('click',()=>{
  const cols=['group','rank','method','events','raw_lines','max_hosts','labels','normalized'], out=[cols.join(',')];
  for(const g of selected())g.querySelectorAll('li.rep').forEach((li,i)=>{const d=li.dataset;
    out.push([g.dataset.g,i+1,d.m,d.e,d.r,d.h,d.l,li.querySelector('.line code').textContent].map(csv).join(','));});
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([out.join('\\r\\n')+'\\r\\n'],{type:'text/csv'}));
  a.download='cmdstack-selected.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
});
const order=document.getElementById('order');
if(order)order.addEventListener('change',()=>{const k=order.value, first=groups[0].parentNode, after=document.getElementById('noisehead');
  [...groups].sort((a,b)=>a.dataset[k]-b.dataset[k]).forEach(g=>first.insertBefore(g,after));});
applyFilter(); updateExport();
"""


def _html(summary: dict, groups: pd.DataFrame, noise: pd.DataFrame, reps: dict, stats: pd.DataFrame,
          raw: dict, minority: dict, has_labels: bool, has_hosts: bool, has_status: bool = False,
          has_baseline: bool = False, prior_reviewed: dict | None = None, look: dict | None = None) -> str:
    look = look or {}
    e = html.escape
    fmt = lambda n: f"{int(n):,}"
    plural = lambda n, word: f"{fmt(n)} {word}{'' if int(n) == 1 else 's'}"
    copy_btn = '<button class="cp" type="button" title="Copy to clipboard">copy</button>'
    cards = [("events", summary["events"]), ("distinct lines", summary["distinct_lines"]),
             ("patterns (templates)", summary["templates"]), ("groups", summary["groups"]),
             ("example lines", summary["representative_lines"]), ("noise lines", summary["noise_lines"])]
    if has_labels:
        cards.append(("mixed groups", summary["mixed_groups"]))
    if has_baseline and summary.get("status"):
        st = summary["status"]
        cards = [("new groups", st["new"]), ("changed groups", st["changed"]),
                 ("reviewed, with new lines", st[bl.REVIEWED_NEW]), ("known groups", st["known"]),
                 ("reviewed (hidden)", st["reviewed"])] + cards
    card_html = "".join(f'<div class="card"><b>{fmt(v)}</b><span>{e(k)}</span></div>' for k, v in cards)

    def labels_html(labels: str, majority) -> str:
        parts = [v for v in labels.split(MIXED_SEPARATOR) if v]
        return ", ".join(f'<span class="hot">{e(v)}</span>' if majority and v != majority else e(v) for v in parts)

    def rep_item(method: str, text: str, majority) -> str:
        s = stats.loc[text]
        hosts = int(s["max_hosts"]) if has_hosts else ""
        labels = s["labels"] if has_labels else ""
        bits = [plural(s["events"], "event"), plural(s["raw_lines"], "raw line")]
        if has_hosts:
            bits.append(f"up to {plural(hosts, 'host')}")
        bits.append(f"{METHOD_NAMES.get(method, method)} example")
        if has_labels:
            bits.append(labels_html(labels, majority))
        rows = "".join(
            f'<tr><td class="n">{plural(r["events"], "event")}'
            + (f'<br>{plural(r["hosts"], "host")}' if has_hosts else "")
            + f'</td><td><div class="line"><code>{e(str(r["command_line"])[:RAW_CHARS])}</code>{copy_btn}</div></td></tr>'
            for r in raw.get(text, []))
        more = int(s["raw_lines"])
        raw_title = ("original command line" if more == 1 else
                     f"original command lines (top {min(more, RAW_SHOWN)} of {fmt(more)})")
        return (f'<li class="rep" data-m="{e(method)}" data-e="{int(s["events"])}" data-r="{more}" '
                f'data-h="{hosts}" data-l="{e(str(labels))}">'
                f'<div class="line"><code>{e(text)}</code>{copy_btn}</div>'
                f'<div class="stats">{" · ".join(bits)}</div>'
                f'<details class="raw"><summary>{raw_title}</summary><table>{rows}</table></details></li>')

    def minority_block(gid: int) -> str:
        if gid not in minority:
            return ""
        others, table = minority[gid]
        rows = "".join(f'<tr><td class="n">{plural(r["events"], "event")}<br>{plural(r["raw_lines"], "raw line")}'
                       f'</td><td><div class="line"><code>{e(r["normalized"][:RAW_CHARS])}</code>{copy_btn}</div></td></tr>'
                       for _, r in table.head(MINORITY_SHOWN).iterrows())
        more = len(table) - MINORITY_SHOWN
        note = f'<p class="meta">and {fmt(more)} more in lines.csv.gz</p>' if more > 0 else ""
        names = ", ".join(f'<span class="hot">{e(v)}</span>' for v in others)
        return (f'<div class="minority"><p class="meta">Lines labeled {names} in this group '
                f'({fmt(len(table))}), most events first</p><table>{rows}</table>{note}</div>')

    def hosts_block(row) -> str:
        names = [n for n in str(row.get("host_names") or "").split(", ") if n] if has_hosts else []
        if len(names) <= FEW_HOSTS:
            return ""
        capped = f" (each line keeps up to {fmt(HOST_SAMPLE)} names)" if row.get("hosts_note") == "+" else ""
        return (f'<details class="hosts"><summary>{plural(len(names), "machine")}{capped}; also in '
                f'group_hosts.csv</summary><p><code>{e(", ".join(names))}</code></p></details>')

    def look_block(gid: int) -> str:
        if gid not in look:
            return ""
        rows, total = look[gid]
        body = "".join(
            f'<tr><td class="n">{plural(r["events"], "event")}'
            + (f'<br>{plural(r["hosts"], "host")}' if has_hosts else "")
            + (f'<br>{e(str(r["host_names"]).replace("; ", ", "))}'
               if has_hosts and r.get("host_names") and 0 < int(r["hosts"]) <= FEW_HOSTS else "")
            + f'</td><td><div class="line"><code>{e(str(r["command_line"])[:RAW_CHARS])}</code>{copy_btn}</div>'
            + f'<span class="flag">{e(r["reasons"])}</span>'
            + (f'<div class="stats">decoded: <code>{e(str(r["b64_decoded"])[:300])}</code></div>'
               if r["b64_decoded"] else "")
            + "</td></tr>" for _, r in rows.iterrows())
        more = f'<p class="meta">and {fmt(total - len(rows))} more in lines.csv.gz</p>' if total > len(rows) else ""
        return (f'<div class="look"><p class="meta">Lines worth a look ({fmt(total)}): from temp or user folders, '
                f'encoded, from programs other than the title\'s, or not seen before</p>'
                f'<table>{body}</table>{more}</div>')

    def group_block(row) -> str:
        gid = int(row["group"])
        mixed = has_labels and bool(row.get("mixed"))
        majority = row.get("majority_label") if mixed else None
        items = "".join(rep_item(m, t, majority) for m, t in reps.get(gid, []))
        badge = '<span class="mixed">mixed labels</span>' if mixed else ""
        labels = f' · labels: {labels_html(str(row["labels"]), majority)}' if has_labels else ""
        main_program = str(row["top_programs"]).split(", ")[0]
        hosts = ""
        if has_hosts:
            note = str(row["hosts_note"]) if pd.notna(row["hosts_note"]) else ""
            hosts = (f' · {"at least " if note == "at least" else ""}{fmt(row["hosts"])}{"+" if note == "+" else ""}'
                     f' host{"" if row["hosts"] == 1 else "s"}')
            names = [n for n in str(row.get("host_names") or "").split(", ") if n]
            if 0 < len(names) <= FEW_HOSTS:
                hosts += f' ({e(", ".join(names))})'
        status = str(row.get("status") or "") if has_status else ""
        label, css = status, {bl.REVIEWED_NEW: "renew"}.get(status, status)
        if status == "changed" and pd.notna(row.get("hosts_before")):
            label = f'changed: {fmt(row["hosts_before"])} → {fmt(row["hosts"])} hosts'
        if status == bl.REVIEWED_NEW:
            label = f'reviewed, {plural(row["unseen_warning_lines"], "new line")} to check'
        status_badge = f'<span class="st st-{css}">{e(label)}</span> ' if status else ""
        others = [p for p in str(row.get("other_programs") or "").split(", ") if p]
        also = ""
        if others:
            extra = len(others) - OTHER_PROGRAMS_SHOWN
            also = (f'<span class="also">+ {e(", ".join(others[:OTHER_PROGRAMS_SHOWN]))}'
                    f'{f" and {extra} more programs" if extra > 0 else ""}</span>')
        flags = []
        if row.get("user_writable_lines"):
            flags.append(f'{plural(row["user_writable_lines"], "line")} from temp or user folders')
        if row.get("encoded_lines"):
            flags.append(plural(row["encoded_lines"], "encoded line"))
        flag_html = "".join(f'<span class="flag">{e(f)}</span>' for f in flags)
        return (f'<div class="grp" data-g="{gid}" data-id="{e(str(row["group_id"]))}" data-status="{status}" '
                f'data-p="{e(main_program)}" data-vr="{int(row["volume_rank"])}" '
                f'data-rr="{int(row["rarity_rank"])}"{" data-mixed=1" if mixed else ""}>'
                f'<input type="checkbox" title="Select for export" aria-label="Select group {gid}">'
                f'<details><summary><span class="gid" title="group id for cmdstack review: '
                f'{e(str(row["group_id"]))}">#{gid}</span> '
                f'<code>{e(row["title"][:160])}</code> {also}{status_badge}{badge}{flag_html}'
                f'<span class="meta">{fmt(row["events"])} events · {fmt(row["distinct_lines"])} lines{hosts}'
                f'{labels}</span></summary>'
                f'<p class="meta">Group id {e(str(row["group_id"]))} · Programs: {e(row["top_programs"])} · main pattern covers '
                f'{row["main_template_share"]}% of the group\'s events</p>{hosts_block(row)}{look_block(gid)}{minority_block(gid)}'
                f'<p class="meta">Example lines</p><ol>{items}</ol></details></div>')

    program_counts = (groups["top_programs"].str.split(", ").str[0].value_counts().head(PROGRAM_CHIPS)
                      if not groups.empty else pd.Series(dtype=int))
    chips = "".join(f'<button type="button" class="chip" data-p="{e(p)}">{e(p)} <span>{fmt(n)}</span></button>'
                    for p, n in program_counts.items())
    chips_html = (f'<div class="chips"><span class="meta">Groups by program:</span>'
                  f'<button type="button" class="chip on" data-p="">all</button>{chips}</div>' if chips else "")

    blocks = "".join(group_block(r) for _, r in groups.iterrows()) if not groups.empty else "<p>No groups.</p>"
    noise_rows = "".join(f'<tr><td class="n">{fmt(r["events"])}'
                         + (f'<br>{e(str(r["host_names"]).replace("; ", ", "))}'
                            if r.get("host_names") and 0 < int(r.get("hosts", 0) or 0) <= FEW_HOSTS else "")
                         + ('<br><span class="st st-new">new</span>' if has_baseline and not r["seen_before"] else "")
                         + f'</td><td><div class="line"><code>{e(str(r["command_line"])[:RAW_CHARS])}</code>'
                         f'{copy_btn}</div></td></tr>'
                         for _, r in noise.head(200).iterrows())
    # one line up front, the rest in a "how to read this" box the analyst can open once
    if has_baseline:
        lead = "Start with the new and changed groups at the top: they match nothing in the last run, rarest first."
    elif has_hosts:
        lead = "Start at the top: groups seen on the fewest machines, the fewest times, come first."
    else:
        lead = ("Groups are ranked by event count. Add a host column (--host-column) to put the rarest activity first.")
    if has_hosts and "hosts_note" in groups.columns and groups["hosts_note"].eq("at least").any():
        lead += " Machine counts read 'at least N' because the export has counts, not names."
    guide = []
    if has_hosts:
        guide.append("Order: rarest first (fewest hosts, then fewest events). An attack usually touches few machines "
                     "a few times, while routine software runs everywhere. Rare is not the same as malicious: it is "
                     "a reading order. Use Sort to switch to most events.")
    guide.append("Open a group to see example lines from it. Each example says how many events and machines it "
                 "covers; open it again to see the original command lines behind it.")
    guide.append("Examples are picked three ways: typical (a random draw), unusual (the group's outliers) and "
                 "varied (common lines that differ from each other).")
    if has_baseline:
        guide.append("Status: new means nothing like it was in the last run; changed means it was seen before but "
                     "now runs on many more machines (the badge shows before and after), which can be a tool being "
                     "rolled out or moving through the network; known was seen before; reviewed groups were marked "
                     "by an analyst and are hidden unless Show is set to everything, except when they gain new lines "
                     "from temp or user folders, encoded lines or lines from another program: then they show again "
                     "with those lines listed under Lines worth a look.")
    guide.append("The title is the group's most frequent line, so other programs in the group are named after it, "
                 "and badges count lines run from temp or user folders and encoded lines. Open a group to see "
                 "them under Lines worth a look, before marking it reviewed.")
    history = (summary.get("settings") or {}).get("history")
    folder = f'"{history}"' if history else "<history folder>"
    guide.append(f"To hide routine groups next time: tick them, press Mark reviewed, then run "
                 f"cmdstack review {folder} --import <the downloaded reviewed.json>. Or mark them from the "
                 f"terminal by group id (the id column in groups.csv): cmdstack review {folder} --mark <id> "
                 f'--note "why".')
    if has_labels:
        guide.append("Labels: groups with more than one label come first, and list their lines with the less common "
                     "label, because a lone malicious line can sit inside a mostly benign group.")
    guide.append("Lines that fit no group are listed under Noise at the end. One-off commands land there, so "
                 "read it too.")
    guide_html = ("<details class='guide'><summary>How to read this report</summary><ul>"
                  + "".join(f"<li>{e(t)}</li>" for t in guide) + "</ul></details>")
    mixed_toggle = ('<label><input type="checkbox" id="mixedonly"> mixed groups only</label>' if has_labels else "")
    if has_baseline:
        show_select = ('<label>Show <select id="show"><option value="nc">new and changed</option>'
                       '<option value="nr">all but reviewed</option><option value="all">everything</option>'
                       '</select></label>')
    elif has_status:
        show_select = ('<label>Show <select id="show"><option value="nr">all but reviewed</option>'
                       '<option value="all">everything</option></select></label>')
    else:
        show_select = ""
    prior_json = json.dumps({"reviewed": list((prior_reviewed or {}).values())}).replace("</", "<\\/")
    if has_baseline:
        noise_text = (f"{fmt(summary['new_noise_lines'])} of these lines were not seen in the last run; they come "
                      "first, marked new. One-off commands, including attacker reconnaissance, often land here. "
                      "Up to 200 lines are shown; the full list is in noise.csv.")
    else:
        noise_text = ("Lines that fit no group, most frequent first (up to 200; the full list is in noise.csv). "
                      "One-off commands land here, so read this list too.")
    sort_toggle = ('<label>Sort <select id="order"><option value="rr">rarest first</option>'
                   '<option value="vr">most events</option></select></label>' if has_hosts else "")
    skipped = summary.get("rows_skipped", 0)
    skipped_html = (f'<p class="sub"><strong>Warning: {fmt(skipped)} rows of the export could not be read and are not '
                    f'in this report</strong> (for example a row with more fields than the header).</p>'
                    if skipped else "")
    if summary.get("read_leniently"):
        skipped_html += ('<p class="sub"><strong>Note: the export was not clean CSV and was read leniently</strong>; '
                         'a malformed row is kept with its extra fields dropped, so odd lines may come from that.</p>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>cmdstack report</title>
<style>{_CSS}</style></head><body><main>
<h1>cmdstack report</h1><p class="sub">Generated {e(summary["generated"])} from {e(", ".join(summary["files"]))}. Everything ran locally.</p>
{skipped_html}<div class="cards">{card_html}</div>
<h2>Groups</h2><p class="sub">{e(lead)}</p>{guide_html}
{chips_html}
<div class="tools"><input type="search" id="q" placeholder="Filter groups and noise lines, e.g. powershell, net user, reg add" aria-label="Filter">
{show_select}{sort_toggle}{mixed_toggle}<span class="meta" id="shown"></span>
<button type="button" id="selshown">Select shown</button><button type="button" id="export" disabled>Export selected (0)</button>
<button type="button" id="markrev" disabled>Mark reviewed (0)</button></div>
<script type="application/json" id="prior-reviewed">{prior_json}</script>
{blocks}
<h2 id="noisehead">Noise: lines with no group</h2><p class="sub">{e(noise_text)}</p>
<table id="noise">{noise_rows}</table>
</main><script>{_JS}</script></body></html>"""
