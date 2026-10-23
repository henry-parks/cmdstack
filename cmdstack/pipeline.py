"""The four steps: normalize, embed, group, pick representative lines."""
from __future__ import annotations

import hashlib
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd

from .normalize import normalize, program, template

DEFAULT_MODEL = "all-MiniLM-L6-v2"
PCA_DIMS = 30                # vectors are compressed to this many numbers before grouping (see group())
MMR_CANDIDATES = 2_000       # MMR compares every pair in a group, so very large groups are subsampled first


def default_threads() -> int:
    """Half the logical cores, so a run does not freeze the analyst's machine."""
    return max(1, (os.cpu_count() or 2) // 2)


def limit_threads(threads: int | None):
    """Context manager capping numpy/BLAS and torch threads; nullcontext when threads is None."""
    if not threads:
        return nullcontext()
    try:
        import torch
        torch.set_num_threads(threads)
    except ImportError:
        pass
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=threads)


# ---- step 1: normalize ------------------------------------------------------------------------------------
PARALLEL_MIN_LINES = 50_000  # below this, starting worker processes costs more than it saves
FEATURES = ("non_ascii", "unicode_changed", "whitespace_runs", "caret_count", "b64_decoded")


def _normalize_chunk(args) -> dict[str, list]:
    """Normalize a list of command lines; returns column lists (cheaper to send between processes than dicts)."""
    texts, platform, internal_nets = args
    cols = {name: [] for name in ("normalized",) + FEATURES}
    for text in texts:
        normalized, features = normalize(text, platform, internal_nets)
        cols["normalized"].append(normalized)
        for name in FEATURES:
            cols[name].append(features[name])
    return cols


def normalize_lines(lines: pd.DataFrame, platform: str, internal_nets, workers: int | None = None) -> pd.DataFrame:
    """Add normalized, template, program and the obfuscation columns to the distinct-line table.

    Large tables are split across worker processes, since the regex rules run in pure Python.
    """
    texts = lines["command_line"].tolist()
    workers = workers or default_threads()
    if workers > 1 and len(texts) >= PARALLEL_MIN_LINES:
        from concurrent.futures import ProcessPoolExecutor
        size = -(-len(texts) // (workers * 4))
        chunks = [(texts[i:i + size], platform, list(internal_nets)) for i in range(0, len(texts), size)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_normalize_chunk, chunks))
        cols = {name: [v for part in parts for v in part[name]] for name in parts[0]}
    else:
        cols = _normalize_chunk((texts, platform, internal_nets))
    out = lines.copy()
    for name, values in cols.items():
        out[name] = values
    # template and program depend only on the normalized text, so compute them once per unique line
    unique = out["normalized"].unique()
    out["template"] = out["normalized"].map(dict(zip(unique, map(template, unique))))
    out["program"] = out["normalized"].map(dict(zip(unique, map(program, unique))))
    return out


# ---- step 2: embed ----------------------------------------------------------------------------------------
def load_encoder(model: str = DEFAULT_MODEL, model_dir: str | None = None, device: str | None = None, log=None):
    """Return a function texts -> unit vectors. model_dir points at a pre-downloaded copy (air-gapped use).

    device: "auto" (GPU when PyTorch can see one, else CPU), "cuda" (fail if no GPU) or "cpu".
    """
    import logging
    if log:                                      # the import alone takes 10 to 20 seconds, and the first run downloads
        log("loading the language model (the first run downloads it, about 90 MB)")
    # PyTorch warns on Windows and macOS at import that its distributed redirects are unsupported; cmdstack
    # never uses them, and the warning reads like an error in the run log
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
    from sentence_transformers import SentenceTransformer

    try:
        import torch
        has_gpu = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_gpu else None
        cuda_build = torch.version.cuda is not None
    except ImportError:
        has_gpu, gpu_name, cuda_build = False, None, False
    if device in (None, "auto"):
        device = "cuda" if has_gpu else "cpu"
    elif device == "cuda" and not has_gpu:
        hint = ("this PyTorch build has no CUDA support; install a CUDA build from https://pytorch.org"
                if not cuda_build else "PyTorch cannot see a GPU; check the NVIDIA driver")
        raise SystemExit(f"cmdstack: --device cuda requested, but {hint}")
    if log:
        log(f"embedding on {'GPU (' + gpu_name + ')' if device == 'cuda' else 'CPU'}")
    if not sys.stderr.isatty():                  # the model loader's own progress bars flood piped or saved logs
        try:
            from huggingface_hub.utils import disable_progress_bars
            from transformers.utils import logging as hf_logging
            disable_progress_bars()
            hf_logging.disable_progress_bar()
        except ImportError:
            pass
    st = SentenceTransformer(model_dir or model, device=device)
    batch = 256 if device == "cuda" else 64

    def encode(texts):
        return st.encode(list(texts), batch_size=batch, show_progress_bar=sys.stderr.isatty(),
                         normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    return encode


def embed(texts: list[str], encode, cache_dir: Path | None = None, model: str = DEFAULT_MODEL) -> np.ndarray:
    """Embed unique texts, reusing a cached result when the same texts and model were embedded before."""
    if cache_dir is not None:
        digest = hashlib.sha256(("\n".join(texts) + "\x00" + model).encode("utf-8", "replace")).hexdigest()[:16]
        cache = Path(cache_dir) / f"embeddings-{digest}.npy"
        if cache.exists():
            return np.load(cache)
    vectors = encode(texts)
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(cache, vectors)
    return vectors


# ---- step 3: group ----------------------------------------------------------------------------------------
def group(vectors: np.ndarray, min_group: int = 5, min_samples: int = 3, threads: int | None = None) -> np.ndarray:
    """HDBSCAN on the unit vectors, compressed to PCA_DIMS numbers first. -1 means noise.

    fast_hdbscan on 30 PCA dimensions was chosen on measurements: on the OpTC data it scored 0.937 against the
    behavior templates (scikit-learn's HDBSCAN on all 384 numbers: 0.889) with the same share of malicious lines
    in mostly malicious groups, and it grouped 100,000 points in 13 seconds where scikit-learn took 12 minutes.
    """
    import fast_hdbscan
    from sklearn.decomposition import PCA

    n = len(vectors)
    if n < max(min_group, 2):
        return np.full(n, -1)
    data = vectors
    if vectors.shape[1] > PCA_DIMS:
        # always compress: fast_hdbscan needs an extra package for high-dimensional input, so small inputs
        # (fewer points than PCA_DIMS) are compressed to as many numbers as they have points
        data = PCA(n_components=min(PCA_DIMS, n), random_state=0).fit_transform(vectors)
    data = np.ascontiguousarray(data, dtype=np.float64)
    if threads:
        try:
            import numba
            numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
        except Exception:
            pass
    return np.asarray(fast_hdbscan.HDBSCAN(min_cluster_size=min_group, min_samples=min_samples).fit_predict(data))


# ---- step 4: representative lines -------------------------------------------------------------------------
def _farthest_first(vecs: np.ndarray, k: int) -> list[int]:
    """Start at the line nearest the center, then keep adding the line farthest from those chosen."""
    k = min(k, len(vecs))
    center = vecs.mean(axis=0)
    norm = np.linalg.norm(center)
    center = center / norm if norm else center
    chosen = [int(np.argmax(vecs @ center))]
    nearest = 1 - vecs @ vecs[chosen[0]]
    nearest[chosen] = -np.inf
    while len(chosen) < k:
        nxt = int(np.argmax(nearest))
        chosen.append(nxt)
        nearest = np.minimum(nearest, 1 - vecs @ vecs[nxt])
        nearest[chosen] = -np.inf
    return chosen


def _mmr(vecs: np.ndarray, k: int, typical_weight: float = 0.7) -> list[int]:
    """Balance how typical a line is against how different it is from the lines already picked."""
    k = min(k, len(vecs))
    center = vecs.mean(axis=0)
    norm = np.linalg.norm(center)
    center = center / norm if norm else center
    relevance = vecs @ center
    chosen = [int(np.argmax(relevance))]
    max_sim = vecs @ vecs[chosen[0]]
    while len(chosen) < k:
        score = typical_weight * relevance - (1 - typical_weight) * max_sim
        score[chosen] = -np.inf
        nxt = int(np.argmax(score))
        chosen.append(nxt)
        max_sim = np.maximum(max_sim, vecs @ vecs[nxt])
    return chosen


def representatives(labels: np.ndarray, vectors: np.ndarray, texts: list[str], per_method: int = 4,
                    max_shown: int = 8, seed: int = 42) -> dict[int, list[tuple[str, str]]]:
    """For each group: up to max_shown distinct (method, normalized line) picks.

    Three methods, per_method lines each: a random draw, farthest-first and maximal marginal relevance.
    Picks that normalize to the same text are merged; slots fill from random, then farthest-first, then MMR.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for gid in sorted(set(labels.tolist()) - {-1}):
        pos = np.flatnonzero(labels == gid)
        vecs = vectors[pos]
        picks = [("random", int(i)) for i in rng.choice(len(pos), size=min(per_method, len(pos)), replace=False)]
        picks += [("farthest_first", i) for i in _farthest_first(vecs, per_method)]
        sub = np.arange(len(pos))
        if len(pos) > MMR_CANDIDATES:
            sub = np.sort(rng.choice(len(pos), size=MMR_CANDIDATES, replace=False))
        picks += [("mmr", int(sub[i])) for i in _mmr(vecs[sub], per_method)]
        seen, shown = set(), []
        for method, i in picks:
            text = texts[pos[i]]
            if text not in seen:
                seen.add(text)
                shown.append((method, text))
        out[int(gid)] = shown[:max_shown]
    return out


# ---- the whole run ----------------------------------------------------------------------------------------
def run(lines: pd.DataFrame, encode, platform: str = "windows", internal_nets=(), unit: str = "capped",
        min_group: int = 5, max_shown: int = 8, seed: int = 42, cache_dir: Path | None = None,
        model: str = DEFAULT_MODEL, threads: int | None = None, log=lambda msg: None) -> dict:
    """Run all four steps on a distinct-line table (from io.read_events). Returns a dict of results.

    unit="distinct" groups one dot per distinct raw line, as the write-up did. unit="capped" (default) does the
    same but keeps at most min_group dots per normalized line, which bounds the work at scale with nearly the
    same groups. unit="normalized" keeps one dot per normalized line; it is fastest but loses density and is
    not recommended. Embedding cost is the same in every mode, because only unique normalized lines are embedded.
    """
    with limit_threads(threads):
        log(f"normalizing {len(lines):,} distinct lines")
        lines = normalize_lines(lines, platform, internal_nets, workers=threads)
        texts = sorted(lines["normalized"].unique().tolist())
        log(f"embedding {len(texts):,} unique normalized lines")
        vectors = embed(texts, encode, cache_dir, model)
        index = {t: i for i, t in enumerate(texts)}

        if unit == "distinct":
            point_text = lines["normalized"].tolist()
        elif unit == "capped":
            # one dot per distinct line, but at most min_group dots per normalized line: repeats keep dense
            # behaviors dense (as in the write-up) while the number of points stays bounded at scale
            variants = lines["normalized"].value_counts()
            point_text = [t for t in texts for _ in range(min(int(variants[t]), min_group))]
        else:
            point_text = texts
        point_vecs = vectors[[index[t] for t in point_text]]
        log(f"grouping {len(point_text):,} points")
        labels = group(point_vecs, min_group=min_group, threads=threads)

        if unit == "distinct":
            lines["group"] = labels
        else:
            # identical vectors always share a label, so each normalized line maps to one group
            lines["group"] = lines["normalized"].map(dict(zip(point_text, labels.tolist())))
        log(f"picking representative lines for {len(set(labels.tolist()) - {-1}):,} groups")
        reps = representatives(labels, point_vecs, point_text, max_shown=max_shown, seed=seed)
        centroids = group_centroids(lines, vectors, index)
    return {"lines": lines, "representatives": reps, "n_points": len(point_text), "unit": unit,
            "centroids": centroids, "model": model, "texts": texts, "vectors": vectors}


def group_centroids(lines: pd.DataFrame, vectors: np.ndarray, index: dict) -> dict[int, np.ndarray]:
    """Each group's center: the unit-length mean of its unique normalized lines' vectors (used to match runs)."""
    pairs = lines.loc[lines["group"] >= 0, ["group", "normalized"]].drop_duplicates()
    out = {}
    for gid, texts in pairs.groupby("group")["normalized"]:
        center = vectors[[index[t] for t in texts]].mean(axis=0)
        norm = np.linalg.norm(center)
        out[int(gid)] = (center / norm if norm else center).astype(np.float32)
    return out
