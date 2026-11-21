"""p1_t4.py -- P1 Viec 3 (item 12): full mutant study, exactly as frozen in PREREG_06.md v1.0
(sha256 in results/PREREG_06.sha256). Do not change definitions here without PREREG v1.1.

Steps: replay the 36 mutants + the original (fixed '>=') on the 121 videos -> equivalence
filter -> I_R verdicts (explicit-state model checking, primary; corpus, secondary) ->
frontier distance (normalised, polyline through the Pareto points of the 2,730 correct
configurations) -> H4', H5 (with video-level bootstrap, B = 1000) -> H6a/b/c.
Outputs: results/mutants_full.csv, results/p1_t4.json
Usage:   python code/p1_t4.py --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frontier_sweep import (E_ACTIVE, E_IDLE, RATIO, TAUS, build_corpus,  # noqa: E402
                            controller, gate_request)
from mutants import ALL, model_check, run_trace  # noqa: E402
from p0_sanity import monitor_IR  # noqa: E402

TOPIC_DIR = Path(__file__).resolve().parent.parent
T0, W0, R0 = 15, 4, 5
DELTA = 0.1062                     # PREREG sec. 2 (median adjacent-tau distance)
SHIP = "M1a"


# ------------------------------------------------------------------ per-video statistics
def per_video(o: np.ndarray, c: dict, R: int, light: bool = False) -> dict:
    """Per-video sums: frames, opens, events, misses, D>=R events/misses, coverage, viol."""
    vs = c["vstart"]
    nv = len(vs) - 1
    frames = np.diff(vs)
    cs = np.concatenate([[0], np.cumsum(o)])
    opens = cs[vs[1:]] - cs[vs[:-1]]
    on, D, vid = c["ev_on"], c["ev_d"], c["ev_vid"]
    hit = cs[on + D] - cs[on] > 0
    ev = np.bincount(vid, minlength=nv)
    miss = np.bincount(vid, weights=~hit, minlength=nv)
    if light:
        return {"frames": frames, "opens": opens, "ev": ev, "miss": miss}
    # coverage (mAP-proxy): frames of the event at/after its first OPEN
    cov = np.zeros(len(on))
    for i in np.flatnonzero(hit):
        w = np.flatnonzero(o[on[i]:on[i] + D[i]])
        cov[i] = D[i] - w[0]
    dge = D >= R
    ev_ge = np.bincount(vid, weights=dge, minlength=nv)
    miss_ge = np.bincount(vid, weights=(~hit) & dge, minlength=nv)
    covs = np.bincount(vid, weights=cov, minlength=nv)
    tots = np.bincount(vid, weights=D, minlength=nv)
    viol = np.zeros(nv, dtype=np.int64)
    for v in range(nv):
        run = 0
        for x in o[vs[v]:vs[v + 1]]:
            run = 0 if x else run + 1
            viol[v] += run >= R
    return {"frames": frames, "opens": opens, "ev": ev, "miss": miss, "ev_ge": ev_ge,
            "miss_ge": miss_ge, "cov": covs, "tot": tots, "viol": viol}


def agg(pv: dict, idx: np.ndarray) -> dict:
    s = {k: float(v[idx].sum()) for k, v in pv.items()}
    a = s["opens"] / s["frames"]
    return {"a": a, "E": E_IDLE + a * (E_ACTIVE - E_IDLE), "miss_all": s["miss"] / s["ev"],
            "miss_DgeR": s["miss_ge"] / s["ev_ge"] if s["ev_ge"] else float("nan"),
            "n_miss_DgeR": s["miss_ge"], "mAP_proxy": 100 * s["cov"] / s["tot"],
            "viol": s["viol"]}


# ------------------------------------------------------------------------ frontier distance
def pareto(E: np.ndarray, M: np.ndarray) -> np.ndarray:
    order = np.lexsort((M, E))
    keep, best = [], np.inf
    for i in order:
        if M[i] < best:
            keep.append(i)
            best = M[i]
    return np.array(keep)


def seg_dist(p, a, b) -> float:
    ab = b - a
    L = float(ab @ ab)
    t = 0.0 if L == 0 else min(1.0, max(0.0, float((p - a) @ ab) / L))
    return float(np.linalg.norm(p - (a + t * ab)))


def frontier_distance(points: np.ndarray, E: np.ndarray, M: np.ndarray) -> tuple:
    """Normalise by the correct grid; return (distance, dominated_by_pareto) per point."""
    e0, e1, m0, m1 = E.min(), E.max(), M.min(), M.max()
    P = pareto(E, M)
    F = np.column_stack([(E[P] - e0) / (e1 - e0), (M[P] - m0) / (m1 - m0)])
    out = []
    for pE, pM in points:
        p = np.array([(pE - e0) / (e1 - e0), (pM - m0) / (m1 - m0)])
        d = min(seg_dist(p, F[i], F[i + 1]) for i in range(len(F) - 1)) if len(F) > 1 \
            else float(np.linalg.norm(p - F[0]))
        dom = bool(np.any((F[:, 0] <= p[0]) & (F[:, 1] <= p[1])
                          & ((F[:, 0] < p[0]) | (F[:, 1] < p[1]))))
        out.append((d, dom))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    t_start = time.time()
    # topic folder: PREREG at the root, hash in results/; repository: both in prereg/
    pdir = TOPIC_DIR / "prereg" if (TOPIC_DIR / "prereg").is_dir() else None
    prereg = ((pdir or TOPIC_DIR) / "PREREG_06.md").read_bytes()
    prereg_sha = hashlib.sha256(prereg).hexdigest()
    frozen = ((pdir or TOPIC_DIR / "results") / "PREREG_06.sha256").read_text().split()[0]
    assert prereg_sha == frozen, "PREREG_06.md changed after freezing -> write v1.1"
    c = build_corpus()
    nv = len(c["vstart"]) - 1
    allv = np.arange(nv)

    # correct grid, per video (for bootstrap)
    grid_pv, grid_keys = [], []
    for tau in TAUS:
        g = gate_request(c["score"], c["vstart"], tau, tau * RATIO, 12, 6)
        for T in range(5, 31):
            for W in range(1, 16):
                o = controller(g, c["valid"], c["vstart"], T, W, True)
                grid_pv.append(per_video(o, c, min(T, W + 1), light=True))
                grid_keys.append((T, W, tau))
    t_grid = time.time() - t_start

    # mutants (per video, frame traces kept for equivalence + monitor)
    vs = c["vstart"]
    traces = {}
    for mid in ["ORIG"] + list(ALL):
        kw = {} if mid == "ORIG" else ALL[mid][2]
        o = np.zeros(len(c["score"]), dtype=bool)
        for v in range(nv):
            o[vs[v]:vs[v + 1]] = run_trace(c["score"][vs[v]:vs[v + 1]],
                                           c["valid"][vs[v]:vs[v + 1]], T0, W0, kw)
        traces[mid] = o
    # sanity: ORIG == numba fixed controller at the deployed point
    g0 = gate_request(c["score"], c["vstart"], 0.55, 0.30, 12, 6)
    assert np.array_equal(traces["ORIG"], controller(g0, c["valid"], vs, T0, W0, True))

    # equivalence filter (PREREG sec. 2)
    rep, cls_of = {}, {}
    for mid in ALL:
        key = traces[mid].tobytes()
        if np.array_equal(traces[mid], traces["ORIG"]):
            cls_of[mid] = "EQ_ORIG"
            continue
        if key in rep:
            cls_of[mid] = rep[key]
        else:
            rep[key] = mid
            cls_of[mid] = mid
    kept = [m for m in ALL if cls_of[m] == m]

    # model checking (primary I_R verdict) + per-mutant metrics
    pv = {m: per_video(traces[m], c, R0) for m in ["ORIG"] + list(ALL)}
    base = agg(pv["ORIG"], allv)
    rows = []
    for mid, (cls, desc, kw) in ALL.items():
        mc = model_check(T0, W0, kw, R0)
        s = agg(pv[mid], allv)
        # monitor (H6b): delay of the first alarm from the start of the first violating window
        delays = []
        for v in np.flatnonzero(pv[mid]["viol"] > 0):
            o = traces[mid][vs[v]:vs[v + 1]]
            al = monitor_IR(o, R0)
            run, start = 0, None
            for t, x in enumerate(o):
                run = 0 if x else run + 1
                if run == R0:
                    start = t - R0 + 1
                    break
            delays.append(int(np.flatnonzero(al)[0] - start + 1))
        rows.append({"mutant": mid, "class": cls, "description": desc,
                     "equiv_class": cls_of[mid], "kept": mid in kept,
                     "mc_violates": mc["violates"], "mc_seconds": round(mc["seconds"], 4),
                     "mc_states": mc["n_states"], "mc_witness": mc.get("witness"),
                     "corpus_violating_windows": int(s["viol"]),
                     "corpus_videos_violating": int((pv[mid]["viol"] > 0).sum()),
                     "a": s["a"], "E": s["E"], "miss_all": s["miss_all"],
                     "n_miss_DgeR": int(s["n_miss_DgeR"]), "mAP_proxy": s["mAP_proxy"],
                     "dE_pct": 100 * (s["E"] - base["E"]) / base["E"],
                     "dmAP_pts": s["mAP_proxy"] - base["mAP_proxy"],
                     "dmiss_pts": 100 * (s["miss_all"] - base["miss_all"]),
                     "monitor_max_delay": max(delays) if delays else None})
    df = pd.DataFrame(rows)

    # frontier distance (point estimate)
    GO = np.array([p["opens"] for p in grid_pv], float)
    GF = np.array([p["frames"] for p in grid_pv], float)
    GMi = np.array([p["miss"] for p in grid_pv], float)
    GEv = np.array([p["ev"] for p in grid_pv], float)

    def grid_EM(w):
        ga = (GO @ w) / (GF @ w)
        return E_IDLE + ga * (E_ACTIVE - E_IDLE), (GMi @ w) / (GEv @ w)
    gE, gM = grid_EM(np.ones(nv))
    dist = frontier_distance(df[["E", "miss_all"]].to_numpy(), gE, gM)
    df["frontier_dist"] = [d for d, _ in dist]
    df["dominated_by_frontier"] = [dm for _, dm in dist]
    for lab, dl in (("half", DELTA / 2), ("delta", DELTA), ("double", 2 * DELTA)):
        df[f"FI_{lab}"] = (df.mc_violates == True) & (df.frontier_dist <= dl)  # noqa: E712
    df["FI_onesided"] = (df.mc_violates == True) & ((df.frontier_dist <= DELTA)  # noqa: E712
                                                    | ~df.dominated_by_frontier)

    # bootstrap (video-level)
    rng = np.random.default_rng(a.seed)
    V = df[df.kept & (df.mc_violates == True)].mutant.tolist()  # noqa: E712
    boot_frac, boot_ship, boot_d = [], [], {m: [] for m in V}
    for _ in range(a.B):
        idx = rng.integers(0, nv, nv)
        bE, bM = grid_EM(np.bincount(idx, minlength=nv).astype(float))
        pts = np.array([[agg(pv[m], idx)["E"], agg(pv[m], idx)["miss_all"]] for m in V])
        dd = [d for d, _ in frontier_distance(pts, bE, bM)]
        for m, d in zip(V, dd):
            boot_d[m].append(d)
        boot_frac.append(float(np.mean([d <= DELTA for d in dd])) if V else float("nan"))
        if SHIP in V:
            boot_ship.append(dd[V.index(SHIP)] <= DELTA)
        elif cls_of.get(SHIP) in V:
            boot_ship.append(dd[V.index(cls_of[SHIP])] <= DELTA)

    # H6c: corpus testing on 20 random videos
    names = list(c["names"])
    order = np.argsort(names)
    pick = order[np.random.default_rng(a.seed).choice(nv, 20, replace=False)]

    def caught(mid, idx, oracle):
        s, b = agg(pv[mid], idx), agg(pv["ORIG"], idx)
        if oracle == "A":
            return s["n_miss_DgeR"] > 0
        return (abs(s["mAP_proxy"] - b["mAP_proxy"]) >= 0.5
                or abs(100 * (s["E"] - b["E"]) / b["E"]) >= 1.0)
    df["test20_oracleA"] = [caught(m, pick, "A") for m in df.mutant]
    df["test20_oracleB"] = [caught(m, pick, "B") for m in df.mutant]
    rng2 = np.random.default_rng(a.seed + 1)
    subsets = [rng2.choice(nv, 20, replace=False) for _ in range(1000)]
    rate_A = {m: float(np.mean([caught(m, s, "A") for s in subsets])) for m in V}
    rate_B = {m: float(np.mean([caught(m, s, "B") for s in subsets])) for m in V}
    df.to_csv(TOPIC_DIR / "results" / "mutants_full.csv", index=False)

    Vdf = df[df.mutant.isin(V)]
    n_v = len(V)
    k = int(Vdf.FI_delta.sum())
    corpus_viol = df[df.kept & (df.corpus_violating_windows > 0)]
    ship_row = df[df.mutant == SHIP].iloc[0]
    res = {
        "seed": a.seed, "B": a.B, "prereg_sha256": prereg_sha, "delta": DELTA,
        "n_mutants": len(df), "n_equiv_to_orig": int((df.equiv_class == "EQ_ORIG").sum()),
        "n_eff": len(kept), "equivalence_classes": {m: [x for x in ALL if cls_of[x] == m]
                                                    for m in kept if
                                                    sum(cls_of[x] == m for x in ALL) > 1},
        "equiv_to_orig": [m for m in ALL if cls_of[m] == "EQ_ORIG"],
        "n_v_mc": n_v, "violating_kept": V,
        "n_corpus_violating_kept": int(len(corpus_viol)),
        "H4prime": {"k_FI_delta": k, "n_v": n_v, "fraction": k / n_v if n_v else None,
                    "k_FI_half": int(Vdf.FI_half.sum()), "k_FI_double": int(Vdf.FI_double.sum()),
                    "k_FI_onesided": int(Vdf.FI_onesided.sum()),
                    "boot_fraction_ci95": [float(np.percentile(boot_frac, 2.5)),
                                           float(np.percentile(boot_frac, 97.5))],
                    "boot_fraction_mean": float(np.mean(boot_frac)),
                    "PASS": bool(n_v and k / n_v >= 1 / 3)},
        "H5": {"mutant": SHIP, "frontier_dist": float(ship_row.frontier_dist),
               "dominated_by_frontier": bool(ship_row.dominated_by_frontier),
               "boot_share_le_delta": float(np.mean(boot_ship)) if boot_ship else None,
               "PASS": bool(ship_row.frontier_dist <= DELTA and ship_row.mc_violates)},
        "per_mutant_boot_dist_ci95": {m: [float(np.percentile(v, 2.5)),
                                          float(np.percentile(v, 97.5))]
                                      for m, v in boot_d.items()},
        "H6a": {"mc_catches_all_corpus_violating": bool((corpus_viol.mc_violates == True).all()),  # noqa: E712
                "mc_max_seconds": float(df.mc_seconds.max()),
                "mc_max_seconds_violating": float(Vdf.mc_seconds.max()) if n_v else None,
                "n_mc_violating_not_seen_on_corpus": int(((df.mc_violates == True) & df.kept  # noqa: E712
                                                         & (df.corpus_violating_windows == 0)).sum()),
                "PASS": bool((corpus_viol.mc_violates == True).all()  # noqa: E712
                             and df.mc_seconds.max() < 10)},
        "H6b": {"monitor_max_delay_frames": int(corpus_viol.monitor_max_delay.max())
                if len(corpus_viol) else None, "R": R0,
                "PASS": bool(len(corpus_viol) == 0 or corpus_viol.monitor_max_delay.max() <= R0)},
        "H6c": {"videos": [names[i] for i in pick],
                "caught_oracleA": int(Vdf.test20_oracleA.sum()), "n_v": n_v,
                "rate_oracleA": float(Vdf.test20_oracleA.mean()) if n_v else None,
                "caught_oracleB": int(Vdf.test20_oracleB.sum()),
                "rate_oracleB": float(Vdf.test20_oracleB.mean()) if n_v else None,
                "mean_rate_1000_subsets_oracleA": rate_A,
                "mean_rate_1000_subsets_oracleB": rate_B,
                "PASS": bool(n_v and Vdf.test20_oracleA.mean() < 0.5)},
        "baseline_fixed": base,
        "wall_s": {"grid": round(t_grid, 1), "total": round(time.time() - t_start, 1)},
    }
    (TOPIC_DIR / "results" / "p1_t4.json").write_text(json.dumps(res, indent=1, default=str),
                                                     encoding="utf-8")
    cols = ["mutant", "equiv_class", "mc_violates", "corpus_violating_windows", "dE_pct",
            "dmiss_pts", "frontier_dist", "FI_delta", "test20_oracleA", "test20_oracleB"]
    print(df[cols].to_string(index=False))
    print(json.dumps({k: v for k, v in res.items() if k.startswith("H")}, indent=1, default=str))


if __name__ == "__main__":
    main()
