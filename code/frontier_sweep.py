"""frontier_sweep.py -- P1 Viec 3 (item 10): energy-miss frontier of CORRECT configurations.

Correct = fixed comparison '>=' (Theorem 2: R-refresh with R = min(T, W + 1) holds for every
gate sequence). Grid: T in {5..30}, W in {1..15}, tau_open in 7 levels around 0.55 with the
hysteresis ratio kept (tau_close = tau_open * 0.30 / 0.55); watchdog 12/6 as deployed.
Replayed on the 121 videos (n = 166,589 frames, 3,170 events). Per configuration:
  a        activation (fraction of OPEN frames)
  E        Deep3 linear energy model, E = 30.4 + a (290.8 - 30.4) mJ/frame
  miss_all event miss rate over all events
  miss_DgeR miss rate over events with D >= R(T, W)
The Pareto set minimises (E, miss_all). Coordinates are normalised by the range over the
correct grid (min-max), and the frontier is the polyline through the Pareto points sorted by
E (Deep3/T4 definition, CP0 decision 3; theory/THEORY_06.md sec. 4).
Outputs:
  results/frontier_correct.parquet   one row per configuration (+ is_pareto)
  results/frontier_correct.png       scatter + frontier, deployed points marked
  $THS_DATASETS/_derived/06/corpus_flat.npz  (flattened corpus cache for p1_t4.py)
Usage: python code/frontier_sweep.py --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_ref import (CORPUS, DET_ROOT, GATE_TRACE, OUT_ROOT, extract_events,  # noqa: E402
                        load_detections)

TOPIC_DIR = Path(__file__).resolve().parent.parent
E_IDLE, E_ACTIVE = 30.4, 290.8
TAUS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
RATIO = 0.30 / 0.55
CACHE = OUT_ROOT / "corpus_flat.npz"


def build_corpus() -> dict:
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}
    gates = pd.read_parquet(GATE_TRACE)
    score, valid, vstart, ev_on, ev_d, ev_vid, names = [], [], [0], [], [], [], []
    off = 0
    for ds in CORPUS:
        for jp in sorted((DET_ROOT / ds).glob("*.json")):
            key = jp.stem
            video = key.split("__", 1)[1]
            fa, has = load_detections(ds, key)
            gv = gates[(gates.dataset == ds) & (gates.video == video)].sort_values("frame")
            assert len(gv) == len(fa) and gv.frame.iloc[0] == fa[0]
            vi = len(names)
            names.append(f"{ds}/{video}")
            for on, D in extract_events(fa, has):
                ev_on.append(off + on - fa[0])
                ev_d.append(D)
                ev_vid.append(vi)
            score.append(gv.score.to_numpy(np.float64))
            valid.append(gv.valid.to_numpy().astype(np.bool_))
            off += len(fa)
            vstart.append(off)
    c = {"score": np.concatenate(score), "valid": np.concatenate(valid),
         "vstart": np.array(vstart, np.int64), "ev_on": np.array(ev_on, np.int64),
         "ev_d": np.array(ev_d, np.int64), "ev_vid": np.array(ev_vid, np.int64),
         "names": np.array(names)}
    np.savez_compressed(CACHE, **c)
    return c


@njit(cache=True)
def gate_request(score, vstart, tau_open, tau_close, wd_trig, wd_int):
    """g_t = hysteresis OR watchdog, per video (state reset at each video start)."""
    n = len(score)
    g = np.zeros(n, dtype=np.bool_)
    for v in range(len(vstart) - 1):
        st = False
        cc = 0
        for t in range(vstart[v], vstart[v + 1]):
            if score[t] >= tau_open:
                st = True
            elif score[t] <= tau_close:
                st = False
            if st:
                cc = 0
                g[t] = True
            else:
                cc += 1
                g[t] = cc >= wd_trig and cc % wd_int == 0
    return g


@njit(cache=True)
def controller(g, valid, vstart, T, W, ge):
    n = len(g)
    o = np.zeros(n, dtype=np.bool_)
    for v in range(len(vstart) - 1):
        fsl = W + 1
        for t in range(vstart[v], vstart[v + 1]):
            pos = t - vstart[v]
            if not valid[t]:
                fsl += 1
                continue
            cap = fsl >= W if ge else fsl > W
            x = g[t] or (pos > 0 and pos % T == 0) or cap
            o[t] = x
            fsl = 0 if x else fsl + 1
    return o


@njit(cache=True)
def event_miss(o, ev_on, ev_d):
    """missed[i] = no OPEN in [on, on + D)."""
    cs = np.zeros(len(o) + 1, dtype=np.int64)
    for t in range(len(o)):
        cs[t + 1] = cs[t] + (1 if o[t] else 0)
    m = np.zeros(len(ev_on), dtype=np.bool_)
    for i in range(len(ev_on)):
        m[i] = cs[ev_on[i] + ev_d[i]] - cs[ev_on[i]] == 0
    return m


def metrics(o, c, R) -> dict:
    a = float(o.mean())
    miss = event_miss(o, c["ev_on"], c["ev_d"])
    dge = c["ev_d"] >= R
    return {"a": a, "E": E_IDLE + a * (E_ACTIVE - E_IDLE), "miss_all": float(miss.mean()),
            "miss_DgeR": float(miss[dge].mean()) if dge.any() else float("nan"),
            "n_miss": int(miss.sum()), "n_DgeR": int(dge.sum()), "n_miss_DgeR": int(miss[dge].sum())}


def pareto_mask(E: np.ndarray, M: np.ndarray) -> np.ndarray:
    idx = np.lexsort((M, E))
    keep = np.zeros(len(E), dtype=bool)
    best = np.inf
    for i in idx:
        if M[i] < best:
            keep[i] = True
            best = M[i]
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="unused (deterministic); recorded")
    a = ap.parse_args()
    t0 = time.time()
    c = build_corpus()
    rows = []
    for tau in TAUS:
        g = gate_request(c["score"], c["vstart"], tau, tau * RATIO, 12, 6)
        for T in range(5, 31):
            for W in range(1, 16):
                o = controller(g, c["valid"], c["vstart"], T, W, True)
                R = min(T, W + 1)
                rows.append({"T": T, "W": W, "tau_open": tau, "tau_close": round(tau * RATIO, 4),
                             "R": R, **metrics(o, c, R)})
    df = pd.DataFrame(rows)
    df["is_pareto"] = pareto_mask(df.E.to_numpy(), df.miss_all.to_numpy())
    df.attrs["seed"] = a.seed
    df.to_parquet(TOPIC_DIR / "results" / "frontier_correct.parquet", index=False)
    # deployed points (for the figure only)
    g = gate_request(c["score"], c["vstart"], 0.55, 0.30, 12, 6)
    dep = {k: metrics(controller(g, c["valid"], c["vstart"], 15, 4, ge), c, 5)
           for k, ge in (("fixed_ge", True), ("deployed_gt", False))}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(df.E, 100 * df.miss_all, s=4, c="#b8b8b8", label=f"correct configs (n = {len(df)})")
    p = df[df.is_pareto].sort_values("E")
    ax.plot(p.E, 100 * p.miss_all, "-o", ms=3, c="#1f5fa8", label=f"Pareto frontier (n = {len(p)})")
    ax.scatter([dep["fixed_ge"]["E"]], [100 * dep["fixed_ge"]["miss_all"]], marker="s", c="#2a9d4b",
               zorder=5, label="fixed '>=' (15, 4, 0.55)")
    ax.scatter([dep["deployed_gt"]["E"]], [100 * dep["deployed_gt"]["miss_all"]], marker="X", s=60,
               c="#c0392b", zorder=5, label="shipped '>' (15, 4, 0.55)")
    ax.set_xlabel("energy E (mJ/frame, Deep3 model)")
    ax.set_ylabel("event miss rate (%, all events)")
    ax.set_title("121 videos, 3,170 events; T in 5..30, W in 1..15, 7 tau levels")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(TOPIC_DIR / "results" / "frontier_correct.png", dpi=160)
    info = {"seed": a.seed, "n_configs": len(df), "n_pareto": int(df.is_pareto.sum()),
            "n_frames": int(len(c["score"])), "n_events": int(len(c["ev_on"])),
            "n_videos": int(len(c["vstart"]) - 1), "deployed_points": dep,
            "E_range": [float(df.E.min()), float(df.E.max())],
            "miss_range": [float(df.miss_all.min()), float(df.miss_all.max())],
            "wall_s": round(time.time() - t0, 1)}
    (TOPIC_DIR / "results" / "frontier_correct_info.json").write_text(json.dumps(info, indent=1),
                                                                      encoding="utf-8")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
