"""Run the benchmark grid and write one parquet file per (view, n) cell.

    python -I src/lsb/run_experiment.py --reps 20 --jobs 12                 # reps 0..19
    python -I src/lsb/run_experiment.py --rep-start 20 --reps 80 --jobs 12  # reps 20..99
    python -I src/lsb/run_experiment.py --views "assist17__d20" --n 50 100 --reps 3 --jobs 2

Cells already on disk are skipped, so the command can be re-run after an interruption.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one BLAS thread per worker process

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
N_LIST = (50, 100, 200, 500)


def _one(view, n, rep, learned, models, lomo=False):
    import warnings

    warnings.filterwarnings("ignore")
    from protocol import run_lomo_rep, run_rep

    if lomo:
        return run_lomo_rep(view, n, rep, learned=learned, models=models)
    return run_rep(view, n, rep, learned=learned, models=models)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="*", default=["*"], help="glob patterns on view names")
    ap.add_argument("--exclude", nargs="*", default=[], help="glob patterns to skip")
    ap.add_argument("--n", nargs="*", type=int, default=list(N_LIST))
    ap.add_argument("--reps", type=int, default=30, help="number of repetitions in this pass")
    ap.add_argument("--rep-start", type=int, default=0,
                    help="first repetition index (passes with different ranges can be combined)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--no-learned", action="store_true")
    ap.add_argument("--no-models", action="store_true")
    ap.add_argument("--lomo", action="store_true",
                    help="OULAD leave-one-module-out: train on the other modules")
    ap.add_argument("--pool-ratio", type=float, default=1.5,
                    help="a cell is run only if the candidate pool has at least ratio*n learners")
    a = ap.parse_args()

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(PROC / "views_index.csv")
    names = [v for v in idx["name"] if any(fnmatch.fnmatch(v, p) for p in a.views)
             and not any(fnmatch.fnmatch(v, p) for p in a.exclude)]
    reps = range(a.rep_start, a.rep_start + a.reps)
    tag = f"r{reps[0]:03d}-{reps[-1]:03d}"
    cells = []
    for v in names:
        pool = int(idx.loc[idx["name"] == v, "n_pool"].iloc[0])
        for n in a.n:
            if pool >= a.pool_ratio * n and not (out / f"{v}__n{n}__{tag}.parquet").exists():
                cells.append((v, n))
    # big, slow cells first so the progress estimate is conservative
    cells.sort(key=lambda c: -c[1] * int(c[0].split("__d")[1]))
    print(f"{len(cells)} cells x reps {reps[0]}..{reps[-1]}, {a.jobs} workers", flush=True)
    t0 = time.time()
    with Parallel(n_jobs=a.jobs, batch_size=1) as par:
        for i, (v, n) in enumerate(cells, 1):
            res = par(delayed(_one)(v, n, r, not a.no_learned, not a.no_models, a.lomo)
                      for r in reps)
            rows = [r for rr, _ in res for r in rr]
            diag = [dg for _, dg in res]
            pd.DataFrame(rows).to_parquet(out / f"{v}__n{n}__{tag}.parquet")
            pd.DataFrame(diag).to_parquet(out / f"{v}__n{n}__{tag}.diag.parquet")
            el = time.time() - t0
            print(f"[{i}/{len(cells)}] {v} n={n}  elapsed {el/60:.1f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
