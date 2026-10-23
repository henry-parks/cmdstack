# cmdstack

Turn a large export of process command lines into a short report you can read in one sitting, on your own machine.
Lines that do the same thing (the same tool with different users, paths, IDs or addresses) become one group, the
rarest activity comes first, and a rerun next week shows what is new. Nothing leaves the machine.

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

cmdstack normalizes each distinct line (user names, addresses, IDs and temp paths become placeholders), embeds each
unique result with a small local model (all-MiniLM-L6-v2), groups them with HDBSCAN and picks up to eight example
lines per group. The work grows with distinct lines, not events: OpTC's 1.6 million events (39,499 distinct lines)
take about 90 seconds on a laptop processor.

## Limits

- cmdstack organizes command lines; it does not decide what is malicious. A single attacker line can sit in a
  routine group, so read the flagged lines before marking a group routine.
- The normalization rules were tuned on public Windows data. The Linux rules (`--platform linux`) are experimental.

## Faster on a GPU

Embedding is the slow step, and an NVIDIA GPU speeds it up several-fold: OpTC takes about 18 seconds instead of 90,
and 3.6 million distinct lines take about 2 minutes. cmdstack uses the GPU automatically when PyTorch can see one.
The PyTorch that pip installs on Windows is CPU-only, so install a CUDA build first, using the command for your
system from [pytorch.org](https://pytorch.org/get-started/locally/), then install cmdstack:

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/henry-parks/cmdstack
```

The first lines of each run say whether it is embedding on the GPU or the processor.

## License

MIT
