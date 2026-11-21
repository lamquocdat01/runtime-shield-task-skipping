"""cb_rule_ext.py -- P2 Viec 1 (item 3): machine check of Proposition 2.7 (G1 + ASMAG circuit
breaker with the exact probe schedule) on the extended grid, against the hand proof in
theory/THEORY_06.md sec. 2.5.

Engine `lmax_cb`: same semantics as region_bruteforce.lmax_cb_exact (BYPASS age +1 per frame,
probe of P frames when age % K == 0, G1 stepped only in ACTIVE and PROBE frames with its
frozen fsl, exit only at the end of a probe, entry at any ACTIVE frame; blocking detector:
the d - 1 frames after a dispatch are busy; kappa = 1, b = 0), with one addition: the
refresh phase at frame 0 is either pinned (pos % T == 0, `allphase = 0`, as lmax_cb_exact)
or adversarial (`allphase = 1`, as Theorem 2' and the IMITATOR models). `cb = 0` gives G1
alone. Gamma = 1 + longest non-dispatch run; R* = Gamma + d.

Statements checked (Gamma_G1 = max(d, min(W'+1, T+d-1)), W' = W (>=) or W+1 (>)):
  (a) sufficiency, every (T, W):        d | gcd(K, P)  =>  Gamma_CB = Gamma_G1
  (b) necessity in the cap-bound regime d - 1 <= W' <= T + d - 3:
        d does not divide gcd(K, P)  =>  Gamma_CB >= max(d, min(W' + 1 + e0, T + d - 1))
        > Gamma_G1, with e0 = K mod d if d does not divide K, else (K - P) mod d
  (c) refresh-bound regime W' + 1 >= T + d - 1:  Gamma_CB = Gamma_G1 for every (K, P)
  (d) upper bound (Prop. 2.6):  Gamma_CB <= max(d, min(W' + d, T + d - 1))
  plus the "iff" rule itself (hole <=> d does not divide gcd(K, P)) per regime.
Grid: K in 2..30 with P | K, 1 <= P < K (prompt), plus every 1 <= P < K for K <= 12;
d in 2..8 (block); (T, W) in TW below; both comparisons; both phase semantics.
Outputs: results/region_ext/cb_rule_ext.json (+ cb_rule_ext.parquet, one row per case)
Usage: python code/cb_rule_ext.py --seed 42 [--workers 3]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

TOPIC_DIR = Path(__file__).resolve().parent.parent
OUT = TOPIC_DIR / "results" / "region_ext"
# (T, W): deployed point, cap-bound points, refresh-bound points, cap-shadowed points
TW = [(15, 4), (30, 8), (12, 6), (20, 10), (9, 3), (4, 12), (5, 8), (3, 6), (15, 1), (20, 0)]


@njit(cache=True)
def _succ(s, g, alt, T, W, ge, Dbusy, B, F, M, K, P, cb):
    """One frame from state s with gate g and mode choice alt. Returns (ok, open, next)."""
    m = s % M
    rest = s // M
    bz = rest % B
    rest //= B
    f = rest % F
    r = rest // F
    rf = r + 1 >= T
    r2 = (r + 1) % T
    valid = bz == 0
    if m == 0:
        if alt == 1:
            if cb == 0:
                return False, False, 0
            op = valid                         # entry frame is a bypass frame, age 0
            f2 = f
            m2 = 1
        else:
            cap = (f >= W) if ge == 1 else (f > W)
            op = valid and (g == 1 or rf or cap)
            f2 = 0 if op else min(f + 1, W + 1)
            m2 = 0
    else:
        mm = m - 1
        age = mm // (P + 1)
        pl = mm % (P + 1)
        age = (age + 1) % K
        if pl == 0 and age == 0:
            pl = P
        if pl > 0:                             # probe frame: G1 decides
            cap = (f >= W) if ge == 1 else (f > W)
            op = valid and (g == 1 or rf or cap)
            f2 = 0 if op else min(f + 1, W + 1)
            pl -= 1
            m2 = 0 if (pl == 0 and alt == 1) else 1 + age * (P + 1) + pl
        else:                                  # pure bypass frame
            if alt == 1 or g == 1:
                return False, False, 0
            op = valid
            f2 = f
            m2 = 1 + age * (P + 1)
    bz2 = Dbusy if op else (bz - 1 if bz > 0 else 0)
    return True, op, ((r2 * F + f2) * B + bz2) * M + m2


@njit(cache=True)
def lmax_cb(T, W, ge, d, K, P, allphase, cb):
    F = W + 2
    Dbusy = d - 1 if d >= 2 else 0
    B = Dbusy + 1
    M = 1 + K * (P + 1) if cb == 1 else 1
    N = T * F * B * M
    reach = np.zeros(N, dtype=np.uint8)
    runstart = np.zeros(N, dtype=np.uint8)
    queue = np.empty(N, dtype=np.int64)
    qt = 0
    # state right after the frame-0 dispatch (fsl_0 = W + 1 forces the cap)
    nph = T if allphase == 1 else 1
    for r0 in range(nph):
        j = ((r0 * F + 0) * B + Dbusy) * M + 0
        reach[j] = 1
        runstart[j] = 1
        queue[qt] = j
        qt += 1
    qh = 0
    while qh < qt:
        s = queue[qh]
        qh += 1
        for choice in range(4):
            ok, op, nx = _succ(s, choice % 2, choice // 2, T, W, ge, Dbusy, B, F, M, K, P, cb)
            if not ok:
                continue
            if op:
                runstart[nx] = 1
            if reach[nx] == 0:
                reach[nx] = 1
                queue[qt] = nx
                qt += 1
    # longest non-dispatch path (acyclic: (f, -busy) increases lexicographically)
    L = np.zeros(N, dtype=np.int32)
    conv = False
    for _ in range((W + 3) * (d + 2) + 5):
        changed = False
        for s in range(N):
            if reach[s] == 0:
                continue
            best = 0
            for alt in range(2):
                ok, op, nx = _succ(s, 0, alt, T, W, ge, Dbusy, B, F, M, K, P, cb)
                if ok and not op:
                    v = 1 + L[nx]
                    if v > best:
                        best = v
            if best != L[s]:
                L[s] = best
                changed = True
        if not changed:
            conv = True
            break
    if not conv:
        return -1
    out = 0
    for s in range(N):
        if runstart[s] == 1 and L[s] > out:
            out = L[s]
    return out


def gamma(T, W, ge, d, K, P, allphase, cb) -> int:
    L = int(lmax_cb(T, W, ge, d, K, P, allphase, cb))
    assert L >= 0, "value iteration did not converge"
    return L + 1


def _job(args):
    T, W, ge, d, K, P, ph = args
    g1 = gamma(T, W, ge, d, 2, 1, ph, 0)
    gcb = gamma(T, W, ge, d, K, P, ph, 1)
    return (T, W, ge, d, K, P, ph, g1, gcb)


def cases():
    KP = set()
    for K in range(2, 31):
        for P in range(1, K):
            if K % P == 0 or K <= 12:
                KP.add((K, P))
    for (K, P) in sorted(KP):
        for d in range(2, 9):
            for (T, W) in TW:
                for ge in (1, 0):
                    for ph in (1, 0):
                        yield (T, W, ge, d, K, P, ph)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="unused (exhaustive); recorded")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    # sanity 1: G1 alone (adversarial phase) == closed form of Theorem 2' (block, b = 0)
    for (T, W) in TW:
        for ge in (1, 0):
            for d in range(2, 9):
                Wp = W if ge else W + 1
                assert gamma(T, W, ge, d, 2, 1, 1, 0) == max(d, min(Wp + 1, T + d - 1)), (T, W, d)
    # sanity 2: pinned phase reproduces region_bruteforce.lmax_cb_exact
    from region_bruteforce import lmax_cb_exact
    for (T, W, ge, d, K, P) in [(15, 4, 1, 4, 300, 30), (15, 4, 0, 3, 300, 30),
                                (6, 2, 1, 2, 6, 3), (8, 4, 1, 3, 9, 4), (2, 2, 1, 2, 4, 2)]:
        assert gamma(T, W, ge, d, K, P, 0, 1) == int(lmax_cb_exact(T, W, ge, d, 1, K, P)) + 1
    jobs = list(cases())
    with Pool(a.workers) as pool:
        rows = pool.map(_job, jobs, chunksize=64)
    df = pd.DataFrame(rows, columns=["T", "W", "ge", "d", "K", "P", "allphase", "G_G1", "G_CB"])
    df["Wp"] = df.W + (1 - df["ge"])
    df["gcd"] = [math.gcd(k, p) for k, p in zip(df.K, df.P)]
    df["dvd"] = df.gcd % df.d == 0
    df["hole"] = df.G_CB > df.G_G1
    df["regime"] = np.where(df.Wp + 1 >= df["T"] + df.d - 1, "refresh_bound",
                            np.where(df.Wp >= df.d - 1, "cap_bound", "cap_shadowed"))
    e0 = np.where(df.K % df.d != 0, df.K % df.d, (df.K - df.P) % df.d)
    df["e0"] = e0
    df["lb_b"] = np.maximum(df.d, np.minimum(df.Wp + 1 + e0, df["T"] + df.d - 1))
    df["ub_d"] = np.maximum(df.d, np.minimum(df.Wp + df.d, df["T"] + df.d - 1))
    df["eps_star"] = df.G_CB - np.minimum(df.Wp + 1, df["T"] + df.d - 1).clip(lower=df.d)
    df.to_parquet(OUT / "cb_rule_ext.parquet", index=False)

    res = {"seed": a.seed, "n_cases": int(len(df)), "wall_s": None,
           "grid": {"K": "2..30, P | K (1 <= P < K), plus all 1 <= P < K for K <= 12",
                    "n_KP": int(df[["K", "P"]].drop_duplicates().shape[0]), "d": "2..8 (block)",
                    "TW": TW, "cmp": [">=", ">"], "phase": ["adversarial", "pinned pos % T == 0"]},
           "by_phase": {}}
    for ph, lab in ((1, "adversarial"), (0, "pinned")):
        s = df[df.allphase == ph]
        ent = {"n": int(len(s))}
        ent["a_sufficiency"] = {"n": int(s.dvd.sum()),
                                "n_violations": int((s.dvd & s.hole).sum())}
        cb = s[(s.regime == "cap_bound") & ~s.dvd]
        ent["b_necessity_cap_bound"] = {"n": int(len(cb)),
                                        "n_no_hole": int((~cb.hole).sum()),
                                        "n_below_lower_bound": int((cb.G_CB < cb.lb_b).sum()),
                                        "n_equal_lower_bound": int((cb.G_CB == cb.lb_b).sum())}
        rb = s[s.regime == "refresh_bound"]
        ent["c_refresh_bound"] = {"n": int(len(rb)), "n_holes": int(rb.hole.sum())}
        ent["d_upper_bound"] = {"n": int(len(s)), "n_violations": int((s.G_CB > s.ub_d).sum())}
        ent["iff_rule_by_regime"] = {
            reg: {"n": int(len(g)), "n_rule_mismatch": int((g.hole != ~g.dvd).sum())}
            for reg, g in s.groupby("regime")}
        cs = s[s.regime == "cap_shadowed"]
        ent["cap_shadowed_holes"] = {"n": int(len(cs)), "n_holes": int(cs.hole.sum()),
                                     "n_holes_with_div": int((cs.hole & cs.dvd).sum())}
        # exact excess in the cap-bound regime vs candidate formulas
        cbx = s[(s.regime == "cap_bound")]
        lhs = cbx.eps_star
        cand = {"e0": cbx.e0.where(~cbx.dvd, 0),
                "d - gcd(d,K,P)": cbx.d - np.gcd(np.gcd(cbx.d, cbx.K), cbx.P)}
        ent["eps_star_candidates_cap_bound_uncapped"] = {}
        unc = (cbx.Wp + 1 + (cbx.d - 1)) <= (cbx["T"] + cbx.d - 1)
        for nm, c in cand.items():
            ent["eps_star_candidates_cap_bound_uncapped"][nm] = {
                "n": int(unc.sum()), "n_match": int((lhs[unc] == c[unc]).sum())}
        res["by_phase"][lab] = ent
    dep = df[(df["T"] == 15) & (df.W == 4) & (df.K == 30) & (df.P == 15) & (df.allphase == 1)]
    res["example_T15_W4_K30_P15"] = dep[["ge", "d", "G_G1", "G_CB"]].to_dict("records")
    res["wall_s"] = round(time.time() - t0, 1)
    (OUT / "cb_rule_ext.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res["by_phase"], indent=1))


if __name__ == "__main__":
    main()
