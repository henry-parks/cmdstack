# cmdstack

Group millions of process command lines into a few hundred behaviors, ranked rarest first, on your own machine.

Give cmdstack a week of process execution events (Sysmon event 1, Security event 4688 or an EDR export). Lines that
differ only in user names, paths, IDs or addresses collapse into one group, the groups are ordered by how few machines
ran them, and a rerun next week flags what is new or has spread. Nothing leaves the machine.

The method and its measurements are in [Making Sense of 1.6 Million Process Execution Events, Part 1](https://henryparks.com/research/clustering-command-lines-locally/),
and the notebook that reruns them is in [research-notebooks](https://github.com/henry-parks/research-notebooks/tree/clustering-command-lines-locally-v1.0/posts/clustering-command-lines-locally).

## Install

Python 3.10 or later:

```
pip install git+https://github.com/henry-parks/cmdstack
```

The install includes PyTorch, so use a virtual environment. No GPU is needed. The first run downloads a 90 MB
model; after that cmdstack works offline.

## Try it

The `examples` folder holds two weeks of made-up events from 20 workstations:

```
git clone https://github.com/henry-parks/cmdstack
cd cmdstack/examples
cmdstack week1.csv.gz --history hunts
cmdstack week2.csv.gz --history hunts
```

The second report shows one new group, a backup agent rolled out to every machine (new, but routine), and under
**Noise** 15 one-off reconnaissance commands on WS-013. Mark the routine group so later runs hide it:

```
cmdstack review hunts --mark <group id> --note "backup agent rollout"
```

## Use it on your data

1. **Export** a week of process starts (Sysmon event 1, Security event 4688 or your EDR), aggregated to one row per
   command line and machine with a count. CSV, gzipped CSV and Excel work.
2. **Run** `cmdstack export.csv --history hunts`, with the same history folder every week. The first lines of output
   show which columns were detected; override them with `--cmd-column`, `--host-column` and `--count-column`.
3. **Read** `report.html`: new and changed groups, rarest first, then the **Noise** list of one-off lines. Badges flag
   lines run from temp or user folders, encoded PowerShell (decoded for you) and other programs in a group.
4. **Scope** anything suspicious with `cmdstack search hunts "<command line>"`, which lists similar lines and the
   machines they ran on.
5. **Hand off** `findings.md`, a ticket-sized summary, and mark routine groups with `cmdstack review`.

If your network uses public address ranges internally, add `--internal-nets <ranges>` to every run, or internal
connections will read as internet traffic. `cmdstack --help` lists every option.

## Reading the results

- Rarity is a reading order, not a verdict: an attack usually touches a few machines a few times, while routine
  software runs everywhere.
- Each group's status compares it with the previous run: **new**, **changed** (on many more machines), **known** or
  **reviewed** (hidden). A reviewed group that gains a flagged line is shown again.
- Search scores are similarity, not probability. Start at the default cutoff of 0.80 and move it to where the lines
  stop being the same activity.

On DARPA OpTC, three attack days compared against a benign week produced 12 new groups out of 87, holding 585 of
the 649 attack lines.

## How it works

1. **Normalize.** Regex rules replace high-cardinality, low-information tokens (user names, IPv4 and IPv6
   addresses, GUIDs, SIDs, ports, hashes, hex, long numbers, temp paths) with typed placeholders. Obfuscation
   (non-ASCII, Unicode tricks, whitespace padding, carets, base64) is kept as separate features rather than
   scrubbed, and base64 is decoded.
2. **Embed.** Each unique normalized line is encoded with all-MiniLM-L6-v2, a 22M-parameter sentence transformer,
   into a 384-dimensional unit vector, so a dot product is cosine similarity. Cost scales with unique normalized
   lines, not events, and vectors are cached by content hash.
3. **Cluster.** PCA projects the vectors to 30 dimensions, then HDBSCAN (`fast_hdbscan`, `min_cluster_size=5`,
   `min_samples=3`) finds density-based clusters without a preset k and labels low-density points as noise, which
   the report lists separately. Each normalized line contributes one point per distinct raw variant, capped at 5,
   so common behaviors stay dense while the point count stays bounded. On OpTC this scores an adjusted Rand index
   of 0.94 against the normalized behavior templates (0.89 for scikit-learn HDBSCAN on all 384 dimensions, as in
   the post) and clusters 100,000 points in 13 seconds.
4. **Pick representatives.** Up to eight lines per cluster from three samplers: random, farthest-first traversal
   for coverage, and maximal marginal relevance (weight 0.7) for typical but non-redundant lines.
5. **Rank and compare.** Clusters are sorted by host count, then event count, fewest first. Each cluster's centroid
   is saved, and the next run matches clusters to it at cosine similarity 0.93 or higher; a cluster whose
   normalized lines were at least 80% seen before also counts as known. A known cluster whose host count at least
   doubles (and grows by 5 or more) is marked changed. `cmdstack search` ranks every normalized line by cosine
   similarity to the query.

OpTC's 1.6 million events (39,499 distinct lines) take about 90 seconds on a laptop processor.

## Limits

- cmdstack organizes command lines; it does not decide what is malicious. A single attacker line can sit in a
  routine group, so read the flagged lines before marking a group routine.
- The normalization rules were tuned on public Windows data. The Linux rules (`--platform linux`) are experimental.

## Faster on a GPU

Embedding is the slow step, and an NVIDIA GPU makes it about five times faster. The OpTC run (1.6 million events)
drops from about 90 seconds on a laptop processor to about 18 seconds, and a larger export of 3.6 million distinct
lines takes about 2 minutes. cmdstack uses the GPU automatically when PyTorch can see one.
The PyTorch that pip installs on Windows is CPU-only, so install a CUDA build first, using the command for your
system from [pytorch.org](https://pytorch.org/get-started/locally/), then install cmdstack:

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/henry-parks/cmdstack
```

The first lines of each run say whether it is embedding on the GPU or the processor.

## License

MIT
