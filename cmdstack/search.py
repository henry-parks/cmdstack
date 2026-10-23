"""cmdstack search: every line like an example command line, and the machines it ran on (scoping a finding).

The query is a command line copied from the report. It is normalized and embedded the same way as the run, then
every normalized line in the run is ranked by cosine similarity. Similarity is not a probability: 0.80 is a
starting cutoff from testing (on DARPA OpTC it returned an Empire stager and its launcher variants and nothing
else), so read the ranked list and move --min-similarity to where the matches stop being the same activity.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .normalize import normalize, parse_networks
from .pipeline import DEFAULT_MODEL, load_encoder

INDEX_VECTORS = "search_index.npy"
INDEX_TEXTS = "search_index.txt.gz"
DEFAULT_MIN = 0.80


def write_index(out_dir: Path, texts: list[str], vectors: np.ndarray) -> None:
    """Save each unique normalized line with its vector, so a search never re-embeds the run."""
    np.save(Path(out_dir) / INDEX_VECTORS, vectors.astype(np.float16))
    with gzip.open(Path(out_dir) / INDEX_TEXTS, "wt", encoding="utf-8") as f:
        f.write("\n".join(texts))           # normalized lines never contain line breaks


def _run_folder(folder: Path, run: str | None) -> Path:
    if run:
        path = Path(run) if Path(run).is_dir() else folder / run
    elif (folder / INDEX_VECTORS).exists():
        path = folder
    else:
        runs = sorted(d for d in folder.iterdir() if d.is_dir() and (d / INDEX_VECTORS).exists()) if folder.is_dir() else []
        path = runs[-1] if runs else folder
    if not (path / INDEX_VECTORS).exists():
        raise SystemExit(f"cmdstack search: no search index in {path}; run cmdstack on an export first "
                         "(runs from cmdstack versions before search have no index)")
    return path


def search_main(argv) -> int:
    p = argparse.ArgumentParser(
        prog="cmdstack search", allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Find every line like an example command line (copied from the report) and the machines it "
                    "ran on.",
        epilog="examples:\n"
               "  cmdstack search hunts 'powershell -nop -w hidden -enc SQBFAFgA...'\n"
               "  cmdstack search hunts 'net user admin P@ss /add' --min-similarity 0.9\n"
               "  cmdstack search hunts '<line>' --run 2026-10-12_093000_week41      last week's run\n"
               "In PowerShell, wrap the line in single quotes (write a ' inside it as ''). If the shell drops the\n"
               "line's own double quotes, the scores barely change.")
    p.add_argument("folder", metavar="FOLDER",
                   help="the --history folder; its latest run is searched unless --run names another (a single "
                        "run's output folder works too)")
    p.add_argument("query", metavar="COMMAND_LINE", help="a command line copied from the report, in quotes")
    p.add_argument("--min-similarity", type=float, default=DEFAULT_MIN, metavar="SCORE",
                   help=f"cutoff for a match, from 0 to 1 (default {DEFAULT_MIN}). Raise it if unrelated lines are "
                        "marked yes, lower it if variants you expect are missing")
    p.add_argument("--limit", type=int, default=20, metavar="N",
                   help="lines to print (default 20); the machine list printed at the end and search.csv in the run "
                        "folder have every match (each search replaces search.csv)")
    p.add_argument("--run", metavar="RUN",
                   help="a run folder to search instead of the latest: a path, or its name under FOLDER")
    p.add_argument("--model-dir", metavar="FOLDER",
                   help="folder with a downloaded copy of the language model, for machines without internet")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cpu",
                   help=argparse.SUPPRESS)          # cpu is fast enough for one query line
    args = p.parse_args(argv)
    run = _run_folder(Path(args.folder), args.run)
    settings = json.loads((run / "summary.json").read_text(encoding="utf-8")).get("settings", {})
    if len(args.query.split()) >= 3 and not re.search(r"[\\/.=\-]", args.query):
        print("note: this looks like plain words; search is built for a command line copied from the report")

    with gzip.open(run / INDEX_TEXTS, "rt", encoding="utf-8") as f:
        texts = f.read().split("\n")
    vectors = np.load(run / INDEX_VECTORS).astype(np.float32)
    query = normalize(args.query, settings.get("platform", "windows"),
                      parse_networks(settings.get("internal_nets", "")))[0]
    encode = load_encoder(settings.get("model", DEFAULT_MODEL), args.model_dir, args.device, log=print)
    sim = dict(zip(texts, vectors @ encode([query])[0]))

    lines = pd.read_csv(run / "lines.csv.gz", keep_default_na=False)
    lines["similarity"] = lines["normalized"].map(sim).astype(float).round(3)
    has_hosts = "host_names" in lines.columns
    top = (lines.groupby("normalized")
           .agg(similarity=("similarity", "first"), events=("events", "sum"),
                hosts=("host_names", lambda s: len({h for v in s for h in v.split("; ") if h})) if has_hosts
                else ("events", "size"))
           .sort_values("similarity", ascending=False).head(args.limit))
    print(f"searching {run} ({len(texts):,} normalized lines)\n\n"
          f"similarity  match  {'hosts' if has_hosts else 'lines'}  line")
    for text, r in top.iterrows():
        match = "yes" if r["similarity"] >= args.min_similarity else "   "
        print(f"{r['similarity']:10.3f}  {match:5}  {int(r['hosts']):5d}  {text[:150]}")

    hits = lines[lines["similarity"] >= args.min_similarity]
    out = run / "search.csv"
    if has_hosts:
        rows = hits.assign(host=hits["host_names"].str.split("; ")).explode("host")
        rows = rows[rows["host"] != ""].sort_values(["similarity", "host"], ascending=[False, True])
        rows[["host", "similarity", "events", "command_line", "normalized"]].to_csv(out, index=False)
        hosts = sorted(rows["host"].unique())
        print(f"\n{len(hosts):,} host{'' if len(hosts) == 1 else 's'} with a line at similarity {args.min_similarity} or "
              f"higher: {', '.join(hosts)}")
    else:
        hits.sort_values("similarity", ascending=False)[["similarity", "events", "command_line", "normalized"]] \
            .to_csv(out, index=False)
    variants = hits.sort_values("similarity", ascending=False)["normalized"].unique()
    print(f"\n{len(variants):,} matched variants, to search for across the full time range in your SIEM or EDR:")
    for v in variants[:args.limit]:
        print(f"  {v[:200]}")
    print(f"\nEvery match with its host and score: {out}. Only this export was searched; confirm scope in the "
          "SIEM or EDR.")
    return 0
