"""p1_shield_check.py -- P1 Viec 2 (item 9): numerical checks of Theorem 4 (optimal shield) in
theory/THEORY_06.md sec. 3.

(a) Exhaustive, short traces. For every policy trace pi in {O,C}^n (n <= 12) and R in {2,3,4},
    enumerate EVERY addition set A (every subset of the CLOSED frames of pi). Any only-add
    enforcer -- causal or not -- produces on pi the output pi | A for some A, so this covers
    the whole class (and more). Check: no feasible A (pi | A in I_R) has |A| < |A_lazy(pi)|,
    and |A_lazy(pi)| = nu(pi), the maximum number of pairwise disjoint OPEN-free R-windows
    of pi (Corollary 4.2, packing = covering).
(b) Closed loop. The theorem is stated for an open-loop (oblivious) policy. A policy that
    reads the shielded output can punish the lazy shield: the explicit counterexample of
    Prop. 4.3 is simulated here.
(c) 121 videos, deployed '>' traces (open loop, the logged o_t of the deployed controller):
    extra activation of the lazy shield L vs three valid baseline enforcers
      eager      : force OPEN after R-2 consecutive closed output frames (one frame early)
      random     : with c >= 1 consecutive closed output frames, force OPEN w.p. 1/(R-c)
                   (deadline-uniform, always valid), seeded
      periodic   : OR an absolute periodic OPEN every R frames (t mod R == 0)
    plus the closed-loop lazy shield around the deployed controller (its fsl is reset by the
    shield's OPENs), which is compared with the fixed '>=' controller.
    Video-level bootstrap, B = 1000, seed 42, 95 % percentile CI.
Inputs: _derived/06/traces_gt.parquet, traces_ge.parquet (THS_DATASETS, default ./datasets)
Output: results/p1_shield.json
Usage:  python code/p1_shield_check.py --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_ref import OUT_ROOT  # noqa: E402

TOPIC_DIR = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ enforcers (one trace)
def lazy(o: np.ndarray, R: int) -> np.ndarray:
    out = o.copy()
    c = 0
    for t in range(len(out)):
        if not out[t] and c >= R - 1:
            out[t] = True
        c = 0 if out[t] else c + 1
    return out


def eager(o: np.ndarray, R: int) -> np.ndarray:
    out = o.copy()
    c = 0
    k = max(R - 2, 0)
    for t in range(len(out)):
        if not out[t] and c >= k:
            out[t] = True
        c = 0 if out[t] else c + 1
    return out


def random_valid(o: np.ndarray, R: int, rng: np.random.Generator) -> np.ndarray:
    out = o.copy()
    c = 0
    u = rng.random(len(out))
    for t in range(len(out)):
        if not out[t]:
            if c >= R - 1 or (c >= 1 and u[t] < 1.0 / (R - c)):
                out[t] = True
        c = 0 if out[t] else c + 1
    return out


def periodic(o: np.ndarray, R: int) -> np.ndarray:
    out = o.copy()
    out[np.arange(len(out)) % R == 0] = True
    return out


def violations(o: np.ndarray, R: int) -> int:
    c, v = 0, 0
    for x in o:
        c = 0 if x else c + 1
        v += c >= R
    return v


def nu(o: np.ndarray, R: int) -> int:
    """Max number of pairwise disjoint OPEN-free windows of length R (greedy from left)."""
    cnt, c = 0, 0
    for x in o:
        if x:
            c = 0
        else:
            c += 1
            if c == R:
                cnt += 1
                c = 0
    return cnt


# ------------------------------------------------------------------------------ (a)
def has_run(x: np.ndarray, R: int, n: int) -> np.ndarray:
    """x: int64 bitmasks of CLOSED frames; True where some R consecutive bits are set."""
    y = x.copy()
    for k in range(1, R):
        y &= x >> k
    return y != 0


def part_a(nmax: int, Rs=(2, 3, 4)) -> dict:
    out = {"n_max": nmax, "R": list(Rs), "n_traces": 0, "n_addition_sets": 0,
           "n_traces_where_some_enforcer_beats_lazy": 0, "n_lazy_infeasible": 0,
           "n_lazy_ne_nu": 0, "examples": []}
    for R in Rs:
        for n in range(1, nmax + 1):
            full = (1 << n) - 1
            for pm in range(1 << n):                      # bit t = 1 <=> OPEN at t
                o = np.array([(pm >> t) & 1 for t in range(n)], dtype=bool)
                closed_pos = [t for t in range(n) if not o[t]]
                k = len(closed_pos)
                # all subsets A of the closed positions as bitmasks over frames
                sub = np.arange(1 << k, dtype=np.int64)
                amask = np.zeros(1 << k, dtype=np.int64)
                size = np.zeros(1 << k, dtype=np.int64)
                for j, t in enumerate(closed_pos):
                    bit = (sub >> j) & 1
                    amask |= bit << t
                    size += bit
                closed_after = (~(pm | amask)) & full
                feas = ~has_run(closed_after, R, n)
                L = lazy(o, R)
                aL = int(L.sum() - o.sum())
                out["n_traces"] += 1
                out["n_addition_sets"] += int(len(sub))
                if violations(L, R):
                    out["n_lazy_infeasible"] += 1
                if aL != nu(o, R):
                    out["n_lazy_ne_nu"] += 1
                best = int(size[feas].min()) if feas.any() else None
                if best is not None and best < aL:
                    out["n_traces_where_some_enforcer_beats_lazy"] += 1
                    if len(out["examples"]) < 5:
                        out["examples"].append({"R": R, "trace": "".join("O" if x else "C"
                                                                          for x in o),
                                                "lazy_adds": aL, "best_adds": best})
    return out


# ------------------------------------------------------------------------------ (b)
def closed_loop_counterexample(R: int = 3, k: int = 3, n: int = 600) -> dict:
    """Policy P_k reads the shielded output e: it opens at t iff one of the last k outputs
    closes the pattern C^{R-1} O (i.e. it 'punishes' an OPEN that ends a run of R-1 CLOSED
    frames by opening k frames). Lazy adds exactly at such positions; the enforcer E_early
    (force OPEN after R-2 closed frames) never creates the pattern."""
    def run(enf: str) -> tuple[int, int]:
        e = []
        adds = 0
        for t in range(n):
            # policy decision from output history only
            pol = False
            for j in range(1, k + 1):
                end = t - j
                if end - (R - 1) >= 0 and e[end] and not any(e[end - (R - 1):end]):
                    pol = True
            c = 0
            for x in reversed(e):
                if x:
                    break
                c += 1
            thr = R - 1 if enf == "lazy" else R - 2
            out = pol or c >= thr
            adds += int(out and not pol)
            e.append(out)
        assert violations(np.array(e), R) == 0
        return int(sum(e)), adds
    oL, aL = run("lazy")
    oE, aE = run("early")
    return {"R": R, "k": k, "n_frames": n, "lazy_total_open": oL, "lazy_added": aL,
            "early_total_open": oE, "early_added": aE,
            "lazy_beaten_in_total_open": oE < oL}


# ------------------------------------------------------------------------------ (c)
def closed_loop_g1(h_or_g: np.ndarray, rf_T: int, W: int, R: int, gt: bool) -> np.ndarray:
    """Deployed controller with the lazy shield inside the loop (fsl reset by any OPEN)."""
    n = len(h_or_g)
    o = np.zeros(n, dtype=bool)
    fsl, c = W + 1, 0
    for t in range(n):
        cap = fsl > W if gt else fsl >= W
        x = bool(h_or_g[t]) or (t > 0 and t % rf_T == 0) or cap
        if not x and c >= R - 1:
            x = True
        o[t] = x
        fsl = 0 if x else fsl + 1
        c = 0 if x else c + 1
    return o


def boot_ci(per_video: np.ndarray, weights: np.ndarray, B: int, rng) -> list[float]:
    n = len(per_video)
    stats = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        stats.append(float((per_video[i] * weights[i]).sum() / weights[i].sum()))
    return [round(float(np.percentile(stats, 2.5)), 6), round(float(np.percentile(stats, 97.5)), 6)]


def part_c(R: int, T: int, W: int, B: int, seed: int) -> dict:
    gt = pd.read_parquet(OUT_ROOT / "traces_gt.parquet")
    ge = pd.read_parquet(OUT_ROOT / "traces_ge.parquet")
    rng = np.random.default_rng(seed)
    names = ["lazy", "eager", "random", "periodic", "lazy_closed_loop", "fixed_ge"]
    per = {k: [] for k in names}
    viol = {k: 0 for k in names}
    nfr = []
    for (ds, v), x in gt.groupby(["dataset", "video"], sort=True):
        o = x.o_t.to_numpy()
        base = o.sum()
        outs = {"lazy": lazy(o, R), "eager": eager(o, R),
                "random": random_valid(o, R, rng), "periodic": periodic(o, R),
                "lazy_closed_loop": closed_loop_g1(x.g_t.to_numpy(), T, W, R, gt=True)}
        y = ge[(ge.dataset == ds) & (ge.video == v)].o_t.to_numpy()
        outs["fixed_ge"] = y
        for k in names:
            per[k].append((outs[k].sum() - base) / len(o))
            viol[k] += violations(outs[k], R)
        nfr.append(len(o))
    nfr = np.array(nfr, dtype=float)
    res = {"n_videos": len(nfr), "n_frames": int(nfr.sum()), "R": R, "B": B, "seed": seed,
           "definition": "extra activation = (OPEN after enforcer - OPEN of deployed '>' "
                         "trace) / frames, frame-weighted over videos",
           "enforcers": {}}
    lazy_v = np.array(per["lazy"])
    for k in names:
        arr = np.array(per[k])
        res["enforcers"][k] = {
            "extra_activation": round(float((arr * nfr).sum() / nfr.sum()), 6),
            "ci95": boot_ci(arr, nfr, B, rng),
            "residual_IR_violations": int(viol[k]),
            "diff_vs_lazy": round(float(((arr - lazy_v) * nfr).sum() / nfr.sum()), 6),
            "diff_vs_lazy_ci95": boot_ci(arr - lazy_v, nfr, B, rng),
            "n_videos_where_fewer_opens_than_lazy": int((arr < lazy_v - 1e-12).sum()),
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nmax", type=int, default=12)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--T", type=int, default=15)
    ap.add_argument("--W", type=int, default=4)
    a = ap.parse_args()
    R = min(a.T, a.W + 1)
    t0 = time.time()
    out = {"seed": a.seed}
    out["a_exhaustive"] = part_a(a.nmax)
    out["a_exhaustive"]["wall_s"] = round(time.time() - t0, 1)
    print("(a)", {k: v for k, v in out["a_exhaustive"].items() if k != "examples"})
    out["b_closed_loop_counterexample"] = closed_loop_counterexample()
    print("(b)", out["b_closed_loop_counterexample"])
    out["c_corpus"] = part_c(R, a.T, a.W, a.B, a.seed)
    for k, v in out["c_corpus"]["enforcers"].items():
        print("(c)", k, v)
    (TOPIC_DIR / "results" / "p1_shield.json").write_text(json.dumps(out, indent=1),
                                                         encoding="utf-8")


if __name__ == "__main__":
    main()
