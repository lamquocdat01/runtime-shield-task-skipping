"""region_bruteforce.py -- P1 Viec 1 (item 5b): exhaustive explicit-state computation of the
extended certificate region of the refresh-bounded gate controller (spec/FSM_06.md sec. 1,
extended in theory/THEORY_06.md sec. 2). Written from the spec; no Deep3/ASMAG code.

Extended controller G(T, W, cmp, d, mode, b, kappa, CB). One step = one frame t.
  * gate g_t in {0,1}: adversarial (any score sequence, any learned gate).
  * refresh: absolute, phase-locked to a wall clock. The refresh clock advances kappa = p/q
    units per frame and is due at every multiple of T (plus an adversarial phase phi).
    Frame t > 0 refreshes iff a due instant lies in (kappa(t-1), kappa t]. A refresh that
    falls on a frame that cannot be dispatched (dropped or detector busy) is LOST (same as
    `pos % T == 0` on a missing frame in the harness). kappa = 1, phi = 0 is Deep3.
  * reuse cap: fsl cmp W with cmp in {'>=' (fixed), '>' (deployed)}; fsl_0 = W + 1.
  * detector delay d >= 0 (frames from dispatch to result).
      mode 'pipe' : pipelined detector, no blocking (result of frame t arrives at t + d).
      mode 'block': one synchronous worker; after a dispatch at t the frames t+1 .. t+d-1
                    cannot be dispatched (busy). Busy frames tick fsl like missing frames.
  * frame drops: valid_t = 0 on a dropped frame; a dropped frame is never OPEN, fsl += 1.
    Budget: at most b dropped frames in every window of R consecutive frames.
  * CB (ASMAG persistent-motion circuit breaker, spec sec. 2): per frame ACTIVE or BYPASS,
    switched by the (adversarial) activation signal. In BYPASS the detector is dispatched on
    every dispatchable frame and the gate controller G1 is NOT stepped (its fsl is frozen;
    `run_full=False` in circuit_breaker.py). ACTIVE and PROBE frames step G1.
    Abstraction: switching instants are free; see THEORY_06.md sec. 2.5 for why this is
    exact for the worst-case gap with the real constants (W_cb = K = 300, P = 30).
  OPEN_t (dispatch) = valid_t and not busy_t and ( BYPASS_t or g_t or rf_t or cap_t ).
Certificate with detector delay (latency counted to the RESULT):
  every event of duration >= R gets a result within R - 1 frames of its onset
  <=> every window of R - d consecutive frames contains a dispatch
  <=> the longest run of non-dispatch frames L satisfies L <= R - d - 1.
So the minimal certified R is  R*(params) = L_max(params) + d + 1, and the region is
  K = {(params, R) : R >= R*(params)}  (monotone in R; the drop budget is per R-window).

Two engines:
  (1) RUN engine (numba): drop budget counted per non-dispatch run (reset at each dispatch).
      Exact for the bad-run question by Lemma 2.1 in THEORY_06.md: a violating run has
      length R - d <= R, so it lies inside one R-window; drops elsewhere only constrain the
      adversary. Computes L_max by reachability + longest non-dispatch path (a DAG).
  (2) WINDOW engine (pure Python, small grid): the literal semantics -- budget over every
      sliding window of R frames, no reset -- used to validate (1): it must find a violation
      at R*-1 and none at R*.
Usage:
  python code/region_bruteforce.py --seed 42 [--quick]
Outputs:
  results/region_ext/bruteforce_grid.parquet  (one row per parameter point, R* and L_max)
  results/region_ext/bruteforce_summary.json  (closed-form check per combination, window check)
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

TOPIC_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = TOPIC_DIR / "results" / "region_ext"

# kappa grid: reduced fractions p/q with q <= 4 in [1/2, 2]
KAPPAS = sorted({Fraction(p, q) for q in range(1, 5) for p in range(1, 9)
                 if Fraction(1, 2) <= Fraction(p, q) <= 2})


# ----------------------------------------------------------------------------- RUN engine
@njit(cache=True)
def _step(r, f, bz, u, drop, g, byp, first, T, W, ge, Dbusy, p, Tq):
    """One frame. Returns (open, r', f', bz', u')."""
    rf = (not first) and (r + p >= Tq)
    r2 = r if first else (r + p) % Tq
    valid = (drop == 0) and (bz == 0)
    if byp == 1:
        op = valid
    else:
        cap = (f >= W) if ge == 1 else (f > W)
        op = valid and (g == 1 or rf or cap)
    if op:
        bz2 = Dbusy
    else:
        bz2 = bz - 1 if bz > 0 else 0
    if byp == 1:
        f2 = f
    else:
        f2 = 0 if op else min(f + 1, W + 1)
    u2 = 0 if op else u + drop
    return op, r2, f2, bz2, u2


@njit(cache=True)
def lmax_run(T, W, ge, d, block, b, p, q, cb):
    """Longest reachable run of non-dispatch frames (drop budget b per run)."""
    Tq = T * q
    F = W + 2
    Dbusy = d - 1 if (block == 1 and d >= 2) else 0
    B = Dbusy + 1
    U = b + 1
    N = Tq * F * B * U
    nmode = 2 if cb == 1 else 1

    def idx(r, f, bz, u):
        return ((r * F + f) * B + bz) * U + u

    reach = np.zeros(N, dtype=np.uint8)
    runstart = np.zeros(N, dtype=np.uint8)
    queue = np.empty(N, dtype=np.int64)
    qh = 0
    qt = 0
    # frame 0 (no refresh), from every phase phi, f = W + 1
    best0 = -1   # runs that start at frame 0 (frame 0 not dispatched): handled below
    first_nonopen = np.zeros(N, dtype=np.uint8)
    for phi in range(Tq):
        for drop in range(2 if b >= 1 else 1):
            for g in range(2):
                for byp in range(nmode):
                    op, r2, f2, bz2, u2 = _step(phi, W + 1, 0, 0, drop, g, byp, True,
                                                T, W, ge, Dbusy, p, Tq)
                    j = idx(r2, f2, bz2, u2)
                    if op:
                        runstart[j] = 1
                    else:
                        first_nonopen[j] = 1
                    if reach[j] == 0:
                        reach[j] = 1
                        queue[qt] = j
                        qt += 1
    while qh < qt:
        s = queue[qh]
        qh += 1
        u = s % U
        rest = s // U
        bz = rest % B
        rest //= B
        f = rest % F
        r = rest // F
        for drop in range(2 if u < b else 1):
            for g in range(2):
                for byp in range(nmode):
                    op, r2, f2, bz2, u2 = _step(r, f, bz, u, drop, g, byp, False,
                                                T, W, ge, Dbusy, p, Tq)
                    j = idx(r2, f2, bz2, u2)
                    if op:
                        runstart[j] = 1
                    if reach[j] == 0:
                        reach[j] = 1
                        queue[qt] = j
                        qt += 1
    # longest non-dispatch path by value iteration (DAG, depth <= W + 1 + b + d)
    L = np.zeros(N, dtype=np.int32)
    for _ in range(W + b + d + 8):
        changed = False
        for s in range(N):
            if reach[s] == 0:
                continue
            u = s % U
            rest = s // U
            bz = rest % B
            rest //= B
            f = rest % F
            r = rest // F
            m = 0
            for drop in range(2 if u < b else 1):
                for byp in range(nmode):
                    op, r2, f2, bz2, u2 = _step(r, f, bz, u, drop, 0, byp, False,
                                                T, W, ge, Dbusy, p, Tq)
                    if not op:
                        v = 1 + L[idx(r2, f2, bz2, u2)]
                        if v > m:
                            m = v
            if m != L[s]:
                L[s] = m
                changed = True
        if not changed:
            break
    best = 0
    for s in range(N):
        if runstart[s] == 1 and L[s] > best:
            best = L[s]
        if first_nonopen[s] == 1 and 1 + L[s] > best:
            best = 1 + L[s]
    return best


def rstar(T, W, ge, d, block, b, kappa: Fraction, cb) -> tuple[int, int]:
    L = int(lmax_run(T, W, ge, d, block, b, kappa.numerator, kappa.denominator, cb))
    return L, L + d + 1


# -------------------------------------------------------------------------- WINDOW engine
def bad_window(T, W, ge, d, block, b, kappa: Fraction, cb, R) -> bool:
    """Literal semantics: <= b drops in EVERY window of R frames; violation iff some run of
    R - d consecutive non-dispatch frames. Pure-Python BFS (small grid only)."""
    if R - d <= 0:
        return True
    need = R - d
    p, q = kappa.numerator, kappa.denominator
    Tq = T * q
    Dbusy = d - 1 if (block and d >= 2) else 0
    modes = (0, 1) if cb else (0,)
    # state: (r, f, bz, run, drops) with drops = bitmask of drops in the last R-1 frames
    mask = (1 << (R - 1)) - 1 if R > 1 else 0
    start = [(phi, W + 1, 0, 0, 0, True) for phi in range(Tq)]
    seen = set()
    stack = list(start)
    while stack:
        st = stack.pop()
        if st in seen:
            continue
        seen.add(st)
        r, f, bz, run, dm, first = st
        ndrops = bin(dm).count("1")
        for drop in ((0, 1) if ndrops + 1 <= b else (0,)):
            for g in (0, 1):
                for byp in modes:
                    rf = (not first) and (r + p >= Tq)
                    r2 = r if first else (r + p) % Tq
                    valid = drop == 0 and bz == 0
                    if byp:
                        op = valid
                    else:
                        cap = f >= W if ge else f > W
                        op = valid and (g == 1 or rf or cap)
                    bz2 = Dbusy if op else max(bz - 1, 0)
                    f2 = f if byp else (0 if op else min(f + 1, W + 1))
                    run2 = 0 if op else run + 1
                    if run2 >= need:
                        return True
                    dm2 = ((dm << 1) | drop) & mask
                    nxt = (r2, f2, bz2, run2, dm2, False)
                    if nxt not in seen:
                        stack.append(nxt)
    return False


# ------------------------------------------------------------------------ closed forms
def cf_base(T, W, ge):
    return min(T, W + 1) if ge else min(T, W + 2)


def closed_form(T, W, ge, d, block, b, kappa: Fraction, cb):
    """Closed forms of Theorem 2' (THEORY_06.md sec. 2). Returns None where no closed form is
    claimed. W' = W (>=) or W + 1 (>): the deployed comparison is the fixed one at W + 1."""
    Wp = W if ge else W + 1
    p, q = kappa.numerator, kappa.denominator
    D = max(d, 1) if block else 1          # minimal dispatch spacing
    s = d - 1 if (block and d >= 2) else 0  # busy (shadow) length after a dispatch
    if cb and b == 0 and kappa == 1:
        # G1 resumes from BYPASS with a frozen fsl; the last bypass dispatch may be up to
        # D frames back (Prop. 2.6)
        if not block or d <= 1:
            return cf_base(T, W, ge) + d
        gap_g1 = max(D, min(Wp + 1, T + s))
        gap_cb = max(D, min(Wp + D, T + s))
        return max(gap_g1, gap_cb) + d
    if cb:
        return None
    # Theorem 2' (general, no CB): one-dispatch gap with b drops, shadow s, rate kappa
    return gap_closed(T, Wp, s, b, p, q) + d


def nref(length, T, p, q):
    """Fewest refresh frames among `length` consecutive frames (adversarial phase):
    floor(length * kappa / T), at most `length` (Lemma 2.2)."""
    return min(length, (length * p) // (T * q))


def gap_closed(T, Wp, s, b, p=1, q=1):
    """Closed form of Theorem 2': largest dispatch-to-dispatch gap Gamma.
    A run of n non-dispatch frames after a dispatch costs
      max(0, n - max(Wp, s))            drops past the cap (cap persists, busy is over)
      + nref(min(n, Wp) - s, T, p, q)   refresh frames between shadow end and cap
    and Gamma = 1 + the largest affordable n. Solved in closed form:
      m := max(0, Wp - s),  c := nref(m)
      c <= b : Gamma = max(Wp, s) + 1 + b - c
      c >  b : Gamma = s + 1 + (largest l <= m with nref(l) <= b)
               = s + min(m, ceil((b + 1) T q / p) - 1) + 1   (kappa-spacing of the (b+1)-th
                 refresh; for kappa = 1: s + (b + 1) T)
    """
    m = max(0, Wp - s)
    c = nref(m, T, p, q)
    if c <= b:
        return max(Wp, s) + 1 + b - c
    lmax = max(b, -(-(b + 1) * T * q // p) - 1)   # ceil((b+1)Tq/p) - 1; >= b when kappa >= T
    return s + 1 + min(m, lmax)


# ------------------------------------------------------------------------------- driver
def _job(args):
    T, W, ge, d, block, b, kap, cb = args
    L, Rs = rstar(T, W, ge, d, block, b, kap, cb)
    return (T, W, ge, d, block, b, str(kap), float(kap), cb, L, Rs)


def build_grid(quick: bool):
    Ts = range(1, 31) if not quick else range(1, 13)
    Ws = range(0, 21) if not quick else range(0, 11)
    ds = range(0, 6)
    bs = range(0, 5)
    jobs = []
    for T, W, ge in itertools.product(Ts, Ws, (1, 0)):
        for d in ds:
            for block in ((0, 1) if d >= 2 else (0,)):
                for b in bs:
                    for kap in KAPPAS:
                        jobs.append((T, W, ge, d, block, b, kap, 0))
                # CB: b = 0, kappa = 1
                jobs.append((T, W, ge, d, block, 0, Fraction(1), 1))
    return jobs


COMBOS = {
    # name: predicate on a row (only the listed factors may differ from the Deep3 setting)
    "base": lambda r: r.d == 0 and r.b == 0 and r.kappa == 1 and r.cb == 0,
    "d": lambda r: r.b == 0 and r.kappa == 1 and r.cb == 0,
    "b": lambda r: r.d == 0 and r.kappa == 1 and r.cb == 0,
    "d_b": lambda r: r.kappa == 1 and r.cb == 0,
    "kappa": lambda r: r.d == 0 and r.b == 0 and r.cb == 0,
    "d_kappa_b": lambda r: r.cb == 0,
    "CB": lambda r: r.d == 0 and r.cb == 1,
    "CB_d": lambda r: r.cb == 1,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="unused (exhaustive); recorded")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--window-check", type=int, default=1)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = build_grid(a.quick)
    rstar(15, 4, 1, 0, 0, 0, Fraction(1), 0)  # JIT warm-up
    with Pool(a.workers) as pool:
        rows = pool.map(_job, jobs, chunksize=256)
    df = pd.DataFrame(rows, columns=["T", "W", "ge", "d", "block", "b", "kappa_str", "kappa",
                                     "cb", "L_max", "R_star"])
    wall_grid = time.time() - t0
    # closed-form check
    cf = []
    for row in df.itertuples(index=False):
        kap = Fraction(row.kappa_str)
        cf.append(closed_form(row.T, row.W, row.ge, row.d, row.block, row.b, kap, row.cb))
    df["R_closed"] = pd.array(cf, dtype="Int64")
    df["closed_ok"] = df.R_closed.isna() | (df.R_closed == df.R_star)
    # naive (additive) formula: min(T, W' + 1) + d + b, to show where the region departs
    Wp = df.W + (1 - df["ge"])
    df["R_naive"] = np.minimum(df["T"], Wp + 1) + df.d + df.b
    df.to_parquet(OUT_DIR / "bruteforce_grid.parquet", index=False)

    summary = {"seed": a.seed, "n_points": int(len(df)), "wall_s_grid": round(wall_grid, 1),
               "grid": {"T": [int(df["T"].min()), int(df["T"].max())],
                        "W": [int(df.W.min()), int(df.W.max())], "d": [0, 5], "b": [0, 4],
                        "kappa": [str(k) for k in KAPPAS]},
               "combos": {}}
    for name, pred in COMBOS.items():
        sub = df[df.apply(pred, axis=1)]
        has_cf = sub.R_closed.notna()
        summary["combos"][name] = {
            "n_points": int(len(sub)),
            "n_with_closed_form": int(has_cf.sum()),
            "n_closed_form_mismatch": int((~sub.closed_ok).sum()),
            "n_naive_mismatch": int((sub.R_naive != sub.R_star).sum()),
            "mismatch_head": sub[~sub.closed_ok].head(10).to_dict("records"),
        }
        print(name, summary["combos"][name]["n_points"],
              "cf-mismatch", summary["combos"][name]["n_closed_form_mismatch"],
              "naive-mismatch", summary["combos"][name]["n_naive_mismatch"])

    # literal sliding-window semantics on a small grid
    if a.window_check:
        t1 = time.time()
        wc = {"n_points": 0, "n_mismatch": 0, "mismatch": []}
        small = df[(df["T"] <= 8) & (df.W <= 7) & (df.d <= 3) & (df.b <= 2)
                   & (df.kappa_str.isin(["1", "2/3", "3/2"]))]
        for row in small.itertuples(index=False):
            kap = Fraction(row.kappa_str)
            args = (row.T, row.W, row.ge, row.d, row.block, row.b, kap, row.cb)
            ok = bad_window(*args, row.R_star - 1) and not bad_window(*args, row.R_star)
            wc["n_points"] += 1
            if not ok:
                wc["n_mismatch"] += 1
                if len(wc["mismatch"]) < 20:
                    wc["mismatch"].append(list(map(str, args)) + [int(row.R_star)])
        wc["wall_s"] = round(time.time() - t1, 1)
        wc["grid"] = "T<=8, W<=7, d<=3, b<=2, kappa in {1, 2/3, 3/2}, both cmp, both modes, CB"
        summary["window_semantics_check"] = wc
        print("window check:", {k: v for k, v in wc.items() if k != "mismatch"})
    (OUT_DIR / "bruteforce_summary.json").write_text(json.dumps(summary, indent=1, default=str),
                                                     encoding="utf-8")


if __name__ == "__main__":
    main()


# ------------------------------------------------- CB with the exact ASMAG timing (K, P)
@njit(cache=True)
def lmax_cb_exact(T, W, ge, d, block, K, P):
    """Longest non-dispatch run for G1 + circuit breaker with the EXACT probe timing of
    circuit_breaker.py: BYPASS age +1 per frame; a probe starts when age % K == 0 (not in a
    probe) and lasts P frames during which G1 is stepped; BYPASS can be left only at the end
    of a probe; ACTIVE -> BYPASS is allowed at any frame (the 300-frame activation window can
    always be filled by the adversarial gate, and ACTIVE dwell is unbounded, so this does not
    restrict phases). kappa = 1, b = 0. Mode encoding: m = 0 ACTIVE; m = 1 + age*(P+1) + pl
    for BYPASS with age in [0, K) and probe frames left pl in [0, P] (pl > 0 = in probe)."""
    F = W + 2
    Dbusy = d - 1 if (block == 1 and d >= 2) else 0
    B = Dbusy + 1
    M = 1 + K * (P + 1)
    N = T * F * B * M
    reach = np.zeros(N, dtype=np.uint8)
    runstart = np.zeros(N, dtype=np.uint8)
    queue = np.empty(N, dtype=np.int64)
    qh = 0
    qt = 0
    # frame 0: ACTIVE, f = W + 1, dispatch (cap)
    j = ((0 * F + 0) * B + Dbusy) * M + 0
    reach[j] = 1
    runstart[j] = 1
    queue[qt] = j
    qt += 1
    # successor enumeration shared by BFS and longest path: returns arrays of (open, next)
    nxt = np.empty(8, dtype=np.int64)
    opn = np.empty(8, dtype=np.uint8)
    L = np.zeros(N, dtype=np.int32)
    for phase in range(2):
        it = 0
        while True:
            if phase == 0:
                if qh >= qt:
                    break
                s = queue[qh]
                qh += 1
            else:
                if it >= N:
                    break
                s = it
                it += 1
                if reach[s] == 0:
                    continue
            m = s % M
            rest = s // M
            bz = rest % B
            rest //= B
            f = rest % F
            r = rest // F
            k = 0
            # ---- enumerate (mode decision, gate) ----
            for choice in range(4):
                g = choice % 2
                alt = choice // 2          # ACTIVE: enter bypass?  probe end: exit?
                rf = (r + 1 >= T)
                r2 = (r + 1) % T
                valid = bz == 0
                if m == 0:
                    if alt == 1:           # enter BYPASS at this frame (age 0, not probe)
                        op = valid
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
                    if pl == 0 and age % K == 0:
                        pl = P
                    if pl > 0:             # probe frame: G1 decides
                        cap = (f >= W) if ge == 1 else (f > W)
                        op = valid and (g == 1 or rf or cap)
                        f2 = 0 if op else min(f + 1, W + 1)
                        pl -= 1
                        if pl == 0 and alt == 1:
                            m2 = 0         # EXIT: back to ACTIVE
                        else:
                            m2 = 1 + age * (P + 1) + pl
                    else:
                        if alt == 1:
                            continue
                        op = valid
                        f2 = f
                        m2 = 1 + age * (P + 1)
                if op:
                    bz2 = Dbusy
                else:
                    bz2 = bz - 1 if bz > 0 else 0
                nj = ((r2 * F + f2) * B + bz2) * M + m2
                nxt[k] = nj
                opn[k] = 1 if op else 0
                k += 1
            if phase == 0:
                for i in range(k):
                    if opn[i] == 1:
                        runstart[nxt[i]] = 1
                    if reach[nxt[i]] == 0:
                        reach[nxt[i]] = 1
                        queue[qt] = nxt[i]
                        qt += 1
            else:
                pass
        if phase == 0:
            continue
    # longest non-dispatch path: value iteration (DAG: busy decreases or fsl increases or
    # pure-bypass frames which always dispatch when not busy)
    for _ in range(W + d + 8):
        changed = False
        for s in range(N):
            if reach[s] == 0:
                continue
            m = s % M
            rest = s // M
            bz = rest % B
            rest //= B
            f = rest % F
            r = rest // F
            best = 0
            for alt in range(2):
                rf = (r + 1 >= T)
                r2 = (r + 1) % T
                valid = bz == 0
                if m == 0:
                    if alt == 1:
                        op = valid
                        f2 = f
                        m2 = 1
                    else:
                        cap = (f >= W) if ge == 1 else (f > W)
                        op = valid and (rf or cap)
                        f2 = 0 if op else min(f + 1, W + 1)
                        m2 = 0
                else:
                    mm = m - 1
                    age = mm // (P + 1)
                    pl = mm % (P + 1)
                    age = (age + 1) % K
                    if pl == 0 and age % K == 0:
                        pl = P
                    if pl > 0:
                        cap = (f >= W) if ge == 1 else (f > W)
                        op = valid and (rf or cap)
                        f2 = 0 if op else min(f + 1, W + 1)
                        pl -= 1
                        m2 = 0 if (pl == 0 and alt == 1) else 1 + age * (P + 1) + pl
                    else:
                        if alt == 1:
                            continue
                        op = valid
                        f2 = f
                        m2 = 1 + age * (P + 1)
                if op:
                    continue
                bz2 = bz - 1 if bz > 0 else 0
                v = 1 + L[((r2 * F + f2) * B + bz2) * M + m2]
                if v > best:
                    best = v
            if best != L[s]:
                L[s] = best
                changed = True
        if not changed:
            break
    out = 0
    for s in range(N):
        if runstart[s] == 1 and L[s] > out:
            out = L[s]
    return out


def cb_exact_main(points, K=300, P=30):
    rows = []
    for T, W, ge, d, block in points:
        t0 = time.time()
        L = int(lmax_cb_exact(T, W, ge, d, block, K, P))
        La, _ = rstar(T, W, ge, d, block, 0, Fraction(1), 1)
        Lg, _ = rstar(T, W, ge, d, block, 0, Fraction(1), 0)
        rows.append({"T": T, "W": W, "ge": ge, "d": d, "block": block, "K": K, "P": P,
                     "R_star_exactCB": L + d + 1, "R_star_abstractCB": La + d + 1,
                     "R_star_G1": Lg + d + 1, "wall_s": round(time.time() - t0, 2)})
        print(rows[-1])
    return rows
