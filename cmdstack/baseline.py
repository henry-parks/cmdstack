"""Remember groups between runs: what was seen before, and which groups an analyst marked as reviewed.

Every run writes baseline.json: for each group a stable id, its title and representative lines, host and event
counts, and its center (the mean of its line vectors). A later run given --baseline matches its groups to those
centers by cosine similarity and gives each group a status:

  new       no earlier group is close enough
  known     matches an earlier group
  reviewed  matches an earlier group an analyst marked as reviewed (hidden in the report by default)
  reviewed, new lines   a reviewed group with new lines that carry a warning sign (run from a user-writable
                        folder, encoded, or another program than the title's); shown again
  changed   matches an earlier group, but now runs on many more hosts

Group numbers change between runs, so matching is by meaning, with a line-level backstop: a group whose
normalized lines were mostly seen before counts as known even if its center moved.

The baseline holds no raw command lines beyond the representative (normalized) lines, and no host names.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

BASELINE_FILE: str = "baseline.json"
REVIEWED_FILE: str = "reviewed.json"
# cosine similarity between group centers that counts as the same behavior (tuned on OpTC and BOTSv3 day splits:
# 0.90 missed a malicious netstat group, at no extra benign cost)
MATCH_THRESHOLD: float = 0.93
KNOWN_LINE_SHARE: float = 0.8      # share of a group's normalized lines seen before that also counts as a match
CHANGED_FACTOR: int = 2            # "changed" = host count at least this many times the earlier count ...
CHANGED_MIN_GAIN: int = 5          # ... and at least this many more hosts
VERSION: int = 1
REVIEWED_NEW: str = "reviewed, new lines"   # status of a reviewed group that now has lines not seen before
STATUSES: tuple[str, ...] = ("new", "changed", REVIEWED_NEW, "known", "reviewed")


class ReviewedItem(TypedDict, total=False):
    """A review mark, made with cmdstack review or downloaded from the report's Mark reviewed button."""

    group_id: str
    title: str
    by: str
    at: str
    note: str


class BaselineGroup(TypedDict, total=False):
    """Serialized information retained for one group between runs."""

    id: str
    title: str
    representatives: list[str]
    hosts: int
    events: int
    first_seen: str
    last_seen: str
    reviewed: ReviewedItem
    centroid: list[float]


class BaselineData(TypedDict, total=False):
    """Serialized baseline file contents."""

    version: int
    model: str
    created: str
    groups: list[BaselineGroup]
    lines: dict[str, str]


@dataclass
class PreviousBaseline:
    """Indexed baseline data prepared for comparing one run."""

    present: bool
    groups: list[BaselineGroup]
    lines: dict[str, str]
    groups_by_id: dict[str, BaselineGroup]
    group_ids: list[str]
    centroid_matrix: NDArray[np.float32]


@dataclass
class LineHistory:
    """Line hashes and baseline-seen counts for the current run."""

    unique_lines: pd.DataFrame
    seen_before: pd.Series
    hashes_by_group: dict[int, list[str]]
    unseen_by_group: dict[int, int]


@dataclass
class GroupDecision:
    """Matching and status details calculated for one current group."""

    group_id: str
    status: str
    matched_group_id: str
    similarity: float | None
    seen_line_share: float | None
    hosts_before: int | None


def line_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]


def _new_id(title: str, taken: set[str]) -> str:
    """Deterministic id from the group's title, so rerunning the same data gives the same ids."""
    base = "g" + hashlib.sha1(title.encode("utf-8", "replace")).hexdigest()[:10]
    gid, n = base, 1
    while gid in taken:
        n += 1
        gid = f"{base}-{n}"
    taken.add(gid)
    return gid


def load_baseline(path: str | Path) -> BaselineData:
    """Read baseline.json from a file or from an earlier output folder."""
    p = Path(path)
    if p.is_dir():
        p = p / BASELINE_FILE
    if not p.exists():
        raise SystemExit(f"cmdstack: no baseline found at {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("version") != VERSION:
        raise SystemExit(f"cmdstack: {p} was written by an incompatible cmdstack version")
    return data


def load_reviewed(path: str | Path) -> dict[str, ReviewedItem]:
    """Read reviewed.json (written by cmdstack review or downloaded from the report): group id -> {by, at, note, title}."""
    p = Path(path)
    if p.is_dir():
        p = p / REVIEWED_FILE
    if not p.exists():
        raise SystemExit(f"cmdstack: no reviewed list found at {p}")
    items = json.loads(p.read_text(encoding="utf-8")).get("reviewed", [])
    return {item["group_id"]: item for item in items if item.get("group_id")}


def _prepare_previous_baseline(baseline: BaselineData | None, model: str,
                               reviewed: dict[str, ReviewedItem]) -> PreviousBaseline:
    """Copy, validate, and index the previous run's baseline data."""
    if baseline and baseline.get("model") != model:
        raise SystemExit(f"cmdstack: the baseline was built with model {baseline.get('model')!r}, "
                         f"this run uses {model!r}; their groups cannot be compared")

    # Copy the group dictionaries because review marks are merged into this list.
    old_groups: list[BaselineGroup] = ([dict(group) for group in baseline["groups"]]
                                       if baseline else [])
    old_lines = dict(baseline["lines"]) if baseline else {}
    for group in old_groups:
        if group["id"] in reviewed:
            group["reviewed"] = reviewed[group["id"]]

    group_ids = [group["id"] for group in old_groups]
    centroid_matrix = (np.array([group["centroid"] for group in old_groups], dtype=np.float32)
                       if old_groups else np.zeros((0, 1), dtype=np.float32))
    return PreviousBaseline(
        present=bool(baseline),
        groups=old_groups,
        lines=old_lines,
        groups_by_id={group["id"]: group for group in old_groups},
        group_ids=group_ids,
        centroid_matrix=centroid_matrix,
    )


def _build_line_history(lines: pd.DataFrame, previous: PreviousBaseline) -> LineHistory:
    """Hash each unique normalized line once and summarize its prior appearances."""
    unique_lines = lines[["normalized", "group"]].drop_duplicates("normalized")
    unique_lines = unique_lines.assign(h=unique_lines["normalized"].map(line_hash))

    seen = (dict(zip(unique_lines["normalized"], unique_lines["h"].isin(previous.lines)))
            if previous.present else {})
    seen_before = lines["normalized"].map(seen).fillna(False).astype(bool)
    hashes_by_group = unique_lines.groupby("group")["h"].apply(list).to_dict()
    unseen_by_group = (
        unique_lines[~unique_lines["h"].isin(previous.lines)].groupby("group").size().to_dict()
        if previous.present else {}
    )
    return LineHistory(unique_lines, seen_before, hashes_by_group, unseen_by_group)


def _warning_line_counts(lines: pd.DataFrame, groups: pd.DataFrame,
                         previous: PreviousBaseline) -> dict[int, int]:
    """Count unseen lines carrying a warning sign, grouped by current group id."""
    required = {"user_writable", "b64_decoded", "program"}
    if not previous.present or groups.empty or not required <= set(lines.columns):
        return {}

    other_programs = {
        int(row["group"]): set(filter(None, str(row.get("other_programs") or "").split(", ")))
        for _, row in groups.iterrows()
    }
    candidates = lines.loc[lines["group"] >= 0]
    from_other_program = pd.Series(
        [program in other_programs.get(int(group_id), ())
         for group_id, program in zip(candidates["group"], candidates["program"])],
        index=candidates.index,
    )
    has_warning_sign = (
        candidates["user_writable"].astype(bool)
        | (candidates["encoded"].astype(bool) if "encoded" in candidates.columns else candidates["b64_decoded"] != "")
        | from_other_program
    )
    warning_lines = candidates.loc[has_warning_sign, ["normalized", "group"]]
    warning_lines = warning_lines.drop_duplicates("normalized")
    warning_hashes = warning_lines["normalized"].map(line_hash)
    warning_lines = warning_lines.loc[~warning_hashes.isin(previous.lines)]
    return warning_lines.groupby("group").size().to_dict()


def _match_previous_group(group_id: int, center: NDArray[np.float32],
                          history: LineHistory, previous: PreviousBaseline,
                          threshold: float) -> tuple[str | None, float | None, float | None, bool]:
    """Match a group by centroid, falling back to mostly-seen normalized lines."""
    match, similarity = None, None
    if previous.present and previous.groups:
        scores = previous.centroid_matrix @ center
        best_index = int(np.argmax(scores))
        similarity = float(scores[best_index])
        if similarity >= threshold:
            match = previous.group_ids[best_index]

    hashes = history.hashes_by_group.get(group_id, [])
    known_owners = [previous.lines[line_hash_value]
                    for line_hash_value in hashes if line_hash_value in previous.lines]
    share = len(known_owners) / max(len(hashes), 1) if previous.present else None
    lines_are_known = previous.present and share >= KNOWN_LINE_SHARE

    if lines_are_known and match is None:
        owner_counts = pd.Series([owner for owner in known_owners if owner]).value_counts()
        if len(owner_counts) and owner_counts.index[0] in previous.groups_by_id:
            match = owner_counts.index[0]
    return match, similarity, share, lines_are_known


def _group_status(row: pd.Series, match: str | None, lines_are_known: bool,
                  previous: PreviousBaseline, reviewed: dict[str, ReviewedItem],
                  warning_count: int, group_id: str,
                  covered: set[str] = frozenset()) -> tuple[str, int | None]:
    """Determine a group's status and its prior host count.

    covered: reviewed groups whose mark was made after this same export was last run, so the analyst already saw
    every line in it and none of them should resurface the group.
    """
    status, hosts_before = "", None
    if previous.present:
        if match is None:
            # Lines may have appeared before as noise, without a prior group to compare.
            status = "known" if lines_are_known else "new"
        else:
            old_group = previous.groups_by_id[match]
            hosts = int(row.get("hosts", 0) or 0)
            hosts_before = int(old_group.get("hosts", 0) or 0)
            if ("hosts" in row and hosts >= CHANGED_FACTOR * hosts_before
                    and hosts - hosts_before >= CHANGED_MIN_GAIN):
                status = "changed"
            elif old_group.get("reviewed"):
                status = "reviewed"
            else:
                status = "known"

    if group_id in reviewed and status in ("", "new", "known"):
        status = "reviewed"
    if status == "reviewed" and warning_count and group_id not in covered:
        # Resurface reviewed behavior when unseen lines now carry warning signs.
        status = REVIEWED_NEW
    return status, hosts_before


def _compare_group(row: pd.Series, centroids: dict[int, NDArray[np.float32]],
                   history: LineHistory, previous: PreviousBaseline,
                   reviewed: dict[str, ReviewedItem], warning_counts: dict[int, int],
                   taken_ids: set[str], assigned_ids: set[str],
                   threshold: float, covered: set[str] = frozenset()) -> GroupDecision:
    """Calculate stable identity, match metadata, and status for one group."""
    current_group_id = int(row["group"])
    match, similarity, share, lines_are_known = _match_previous_group(
        current_group_id, centroids[current_group_id], history, previous, threshold
    )
    group_id = (match if match and match not in assigned_ids
                else _new_id(row["title"], taken_ids))
    status, hosts_before = _group_status(
        row, match, lines_are_known, previous, reviewed,
        warning_counts.get(current_group_id, 0), group_id, covered,
    )
    assigned_ids.add(group_id)
    return GroupDecision(
        group_id=group_id,
        status=status,
        matched_group_id=match or "",
        similarity=round(similarity, 3) if similarity is not None else None,
        seen_line_share=round(share, 3) if share is not None else None,
        hosts_before=hosts_before if "hosts" in row else None,
    )


def _add_decisions(groups: pd.DataFrame, decisions: list[GroupDecision],
                   history: LineHistory, warning_counts: dict[int, int],
                   has_baseline: bool) -> pd.DataFrame:
    """Attach comparison results and baseline counts to the group table."""
    groups = groups.copy()
    if groups.empty:
        groups["group_id"] = pd.Series(dtype="str")
        groups["status"] = pd.Series(dtype="str")
        groups["unseen_lines"] = pd.Series(dtype="Int64")
        groups["unseen_warning_lines"] = pd.Series(dtype="Int64")
        groups["hosts_before"] = pd.Series(dtype="Int64")
        groups["matched_group_id"] = pd.Series(dtype="str")
        groups["match_similarity"] = pd.Series(dtype="float64")
        groups["seen_line_share"] = pd.Series(dtype="float64")
        return groups

    groups.insert(1, "group_id", [decision.group_id for decision in decisions])
    groups["status"] = [decision.status for decision in decisions]
    groups["unseen_lines"] = [
        int(history.unseen_by_group.get(int(group_id), 0)) if has_baseline else None
        for group_id in groups["group"]
    ]
    groups["unseen_warning_lines"] = [
        int(warning_counts.get(int(group_id), 0)) if has_baseline else None
        for group_id in groups["group"]
    ]
    groups["hosts_before"] = pd.array(
        [decision.hosts_before for decision in decisions], dtype="Int64"
    )
    groups["matched_group_id"] = [decision.matched_group_id for decision in decisions]
    groups["match_similarity"] = [decision.similarity for decision in decisions]
    groups["seen_line_share"] = [decision.seen_line_share for decision in decisions]
    return groups


def _build_next_baseline(groups: pd.DataFrame, decisions: list[GroupDecision],
                         centroids: dict[int, NDArray[np.float32]],
                         reps: dict[int, list[tuple[str, str]]], history: LineHistory,
                         previous: PreviousBaseline, reviewed: dict[str, ReviewedItem],
                         model: str, now: str) -> BaselineData:
    """Build this run's baseline while retaining groups not observed in this run."""
    records: list[BaselineGroup] = []
    used_group_ids: set[str] = set()
    for (_, row), decision in zip(groups.iterrows(), decisions):
        old_group = previous.groups_by_id.get(decision.matched_group_id)
        if decision.matched_group_id:
            used_group_ids.add(decision.matched_group_id)
        review = reviewed.get(decision.group_id) or (old_group or {}).get("reviewed")
        current_group_id = int(row["group"])
        records.append({
            "id": decision.group_id,
            "title": row["title"],
            "representatives": [text for _, text in reps.get(current_group_id, [])],
            "hosts": int(row["hosts"]) if "hosts" in row and pd.notna(row["hosts"]) else 0,
            "events": int(row["events"]),
            "first_seen": (old_group or {}).get("first_seen", now),
            "last_seen": now,
            "reviewed": review,
            "centroid": [round(float(value), 4) for value in centroids[current_group_id]],
        })

    records.extend(group for group in previous.groups if group["id"] not in used_group_ids)
    current_group_ids = (
        dict(zip(groups["group"], [decision.group_id for decision in decisions]))
        if not groups.empty else {}
    )
    group_id_by_line_hash = dict(zip(
        history.unique_lines["h"],
        history.unique_lines["group"].map(current_group_ids).fillna(""),
    ))
    return {
        "version": VERSION,
        "model": model,
        "created": now,
        "groups": records,
        "lines": {**previous.lines, **group_id_by_line_hash},
    }


def compare(groups: pd.DataFrame, centroids: dict[int, NDArray[np.float32]], lines: pd.DataFrame,
            reps: dict[int, list[tuple[str, str]]], model: str,
            baseline: BaselineData | None = None,
            reviewed: dict[str, ReviewedItem] | None = None,
            threshold: float = MATCH_THRESHOLD,
            covered: set[str] | None = None) -> tuple[pd.DataFrame, BaselineData, pd.Series]:
    """Give each group a stable id and a status; return (groups, new baseline, seen-before flag per line).

    Without a baseline every status is empty, except groups named in the reviewed list. covered: reviewed group ids
    whose mark already covers this export's lines (see _group_status).
    """
    covered = set(covered or ())
    reviewed = dict(reviewed or {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    previous = _prepare_previous_baseline(baseline, model, reviewed)
    history = _build_line_history(lines, previous)
    warning_counts = _warning_line_counts(lines, groups, previous)

    taken_ids = set(previous.group_ids)
    assigned_ids: set[str] = set()
    decisions = [
        _compare_group(row, centroids, history, previous, reviewed, warning_counts,
                       taken_ids, assigned_ids, threshold, covered)
        for _, row in groups.iterrows()
    ]
    groups = _add_decisions(groups, decisions, history, warning_counts, previous.present)
    new_baseline = _build_next_baseline(
        groups, decisions, centroids, reps, history, previous, reviewed, model, now
    )
    return groups, new_baseline, history.seen_before


def write_baseline(baseline: BaselineData, out_dir: str | Path) -> None:
    (Path(out_dir) / BASELINE_FILE).write_text(json.dumps(baseline, separators=(",", ":")), encoding="utf-8")


def save_reviewed(path: str | Path, items: dict[str, ReviewedItem]) -> None:
    """Write reviewed.json in the format the report downloads: {"reviewed": [{group_id, title, by, at, note}]}."""
    rows = sorted(items.values(), key=lambda r: (r.get("at", ""), r["group_id"]))
    Path(path).write_text(json.dumps({"reviewed": rows}, indent=1), encoding="utf-8")
