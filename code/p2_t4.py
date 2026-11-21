"""p2_t4.py -- P2 Viec 2 (item 6): the PREREG v1.1 mutant study (PREREG_06_v1.1.md), reported
side by side with v1.0 (PREREG_06.md). Checks the three frozen hashes before running.

Steps: corpus -> correct grids (G1 as v1.0; G1 + correct circuit breaker for the CB family)
-> traces of ORIG, ORIG_CB and the 87 mutants (36 v1.0 + 51 v1.1) -> equivalence filter on
the union -> MC verdicts (v1.0 engine, mc_g1, mc_cb) -> frontier distances per family ->
delta_tau (both grids), delta_boot (both families) -> v1.0 hypotheses on the v1.0 set (must
reproduce results/p1_t4.json) and on the union -> v1.1 hypotheses H4'', H5'', H6c', H7 ->
exploratory MC with a drop budget b = 1 at R = 6.
Outputs: results/mutants_v11.csv, results/p2_t4.json, results/frontier_correct_cb.parquet
Usage:   python code/p2_t4.py --seed 42
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
                            controller, gate_request, pareto_mask)
from mutants import ALL as ALL_V10, Ctl, model_check, run_trace  # noqa: E402
from mutants_v11 import (ALL_V11, K_CB, P_CB, TH_HI, TH_LO, WIN_CB, Ctl11,  # noqa: E402
                         controller_cb, mc_cb, mc_g1, run_trace11, run_trace_cb)
from p0_sanity import monitor_IR  # noqa: E402
from p1_t4 import agg, frontier_distance, per_video  # noqa: E402

TOPIC_DIR = Path(__file__).resolve().parent.parent
RES = TOPIC_DIR / "results"
T0, W0, R0, TAU0 = 15, 4, 5, 0.55
DELTA_TAU_V10 = 0.1062
SHIP = "M1a"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_hashes() -> dict:
    """Topic folder: PREREG files at the root, hashes in results/. Repository: PREREG files
    and hashes in prereg/, code in code/."""
    pdir = TOPIC_DIR / "prereg" if (TOPIC_DIR / "prereg").is_dir() else None
    hdir = pdir or RES
    v10 = (hdir / "PREREG_06.sha256").read_text().split()[0]
    assert sha((pdir or TOPIC_DIR) / "PREREG_06.md") == v10, "PREREG_06.md changed"
    lines = [ln.split() for ln in (hdir / "PREREG_06_v1.1.sha256").read_text().splitlines()
             if ln.strip() and "*" in ln]
    fro = {ln[1].lstrip("*"): ln[0] for ln in lines}
    for rel, h in fro.items():
        f = TOPIC_DIR / rel
        if not f.exists() and pdir:
            f = pdir / Path(rel).name
        assert sha(f) == h, f"{rel} changed after freezing v1.1"
    return {"v1.0": v10, **fro}


def agg_l(pv: dict, idx: np.ndarray) -> dict:
    """agg() for light per-video stats (frames, opens, ev, miss)."""
    fr, op = float(pv["frames"][idx].sum()), float(pv["opens"][idx].sum())
    a = op / fr
    return {"a": a, "E": E_IDLE + a * (E_ACTIVE - E_IDLE),
            "miss_all": float(pv["miss"][idx].sum()) / float(pv["ev"][idx].sum())}


def id_key(mid: str) -> tuple:
    return ({"M": 0, "N": 1, "C": 2}[mid[0]], mid)


def delta_tau(grid_pts: pd.DataFrame) -> tuple[float, int]:
    """Median normalised distance between adjacent tau levels at fixed (T, W) (v1.0 sec. 2)."""
    E, M = grid_pts.E.to_numpy(), grid_pts.miss_all.to_numpy()
    e = (E - E.min()) / (E.max() - E.min())
    m = (M - M.min()) / (M.max() - M.min())
    g = grid_pts.assign(e=e, m=m).sort_values(["T", "W", "tau_open"])
    d = []
    for _, sub in g.groupby(["T", "W"]):
        x = sub[["e", "m"]].to_numpy()
        d.extend(np.linalg.norm(np.diff(x, axis=0), axis=1).tolist())
    return float(np.median(d)), len(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    t_start = time.time()
    hashes = check_hashes()
    c = build_corpus()
    vs = c["vstart"]
    nv = len(vs) - 1
    allv = np.arange(nv)
    keys = [(T, W, tau) for tau in TAUS for T in range(5, 31) for W in range(1, 16)]

    # ------------------------------------------------------------- correct grids (2 families)
    gpv = {"G1": [], "CB": []}
    rows_cb = []
    for tau in TAUS:
        g = gate_request(c["score"], vs, tau, tau * RATIO, 12, 6)
        for T in range(5, 31):
            for W in range(1, 16):
                R = min(T, W + 1)
                o = controller(g, c["valid"], vs, T, W, True)
                gpv["G1"].append(per_video(o, c, R, light=True))
                ocb = controller_cb(c["score"], c["valid"], vs, T, W, tau, tau * RATIO, 12, 6,
                                    K_CB, P_CB, WIN_CB, TH_HI, TH_LO)
                pv = per_video(ocb, c, R, light=True)
                gpv["CB"].append(pv)
                s = agg_l(pv, allv)
                rows_cb.append({"T": T, "W": W, "tau_open": tau, "tau_close": round(tau * RATIO, 4),
                                "R": R, "a": s["a"], "E": s["E"], "miss_all": s["miss_all"]})
    fcb = pd.DataFrame(rows_cb)
    fcb["is_pareto"] = pareto_mask(fcb.E.to_numpy(), fcb.miss_all.to_numpy())
    fcb.to_parquet(RES / "frontier_correct_cb.parquet", index=False)
    fg1 = pd.read_parquet(RES / "frontier_correct.parquet")
    t_grid = time.time() - t_start

    def mats(fam):
        P = gpv[fam]
        return tuple(np.array([p[k] for p in P], float) for k in ("opens", "frames", "miss", "ev"))
    GM = {f: mats(f) for f in ("G1", "CB")}

    def grid_EM(fam, w):
        GO, GF, GMi, GEv = GM[fam]
        ga = (GO @ w) / (GF @ w)
        return E_IDLE + ga * (E_ACTIVE - E_IDLE), (GMi @ w) / (GEv @ w)

    # ----------------------------------------------------------------------------- traces
    def replay(fn, kw):
        o = np.zeros(len(c["score"]), dtype=bool)
        for v in range(nv):
            sl = slice(vs[v], vs[v + 1])
            o[sl] = fn(c["score"][sl], c["valid"][sl], T0, W0, kw)
        return o
    traces = {"ORIG": replay(run_trace, {}), "ORIG_CB": replay(run_trace_cb, {})}
    g0 = gate_request(c["score"], vs, TAU0, 0.30, 12, 6)
    assert np.array_equal(traces["ORIG"], controller(g0, c["valid"], vs, T0, W0, True))
    cb_num = controller_cb(c["score"], c["valid"], vs, T0, W0, TAU0, 0.30, 12, 6, K_CB, P_CB,
                           WIN_CB, TH_HI, TH_LO)
    n_cb_mismatch = int((cb_num != traces["ORIG_CB"]).sum())
    assert n_cb_mismatch == 0, f"numba CB controller differs from Python on {n_cb_mismatch} frames"
    fam, cls, desc, src = {}, {}, {}, {}
    for mid, (cl, ds, kw) in ALL_V10.items():
        fam[mid], cls[mid], desc[mid], src[mid] = "G1", cl, ds, "v1.0"
        traces[mid] = replay(run_trace, kw)
    for mid, (fm, cl, ds, kw) in ALL_V11.items():
        fam[mid], cls[mid], desc[mid], src[mid] = fm, cl, ds, "v1.1"
        traces[mid] = replay(run_trace_cb if fm == "CB" else run_trace11, kw)
    mids = sorted(fam, key=id_key)
    t_traces = time.time() - t_start

    # ---------------------------------------------------------------- equivalence (union)
    def equiv(ids):
        rep, cl_of = {}, {}
        for mid in sorted(ids, key=id_key):
            orig = "ORIG" if fam[mid] == "G1" else "ORIG_CB"
            if np.array_equal(traces[mid], traces[orig]):
                cl_of[mid] = "EQ_" + orig
                continue
            k = traces[mid].tobytes()
            if k in rep:
                cl_of[mid] = rep[k]
            else:
                rep[k] = mid
                cl_of[mid] = mid
        return cl_of
    cls_union = equiv(mids)
    cls_v10 = equiv([m for m in mids if src[m] == "v1.0"])

    # ---------------------------------------------------------------- MC + per-mutant rows
    pv = {m: per_video(traces[m], c, R0) for m in ["ORIG", "ORIG_CB"] + mids}
    base = {"G1": agg(pv["ORIG"], allv), "CB": agg(pv["ORIG_CB"], allv)}
    rows = []
    for mid in mids:
        if src[mid] == "v1.0":
            kw = ALL_V10[mid][2]
            mc = model_check(T0, W0, kw, R0)
            mcb = mc_g1(Ctl(T0, W0, kw), 6, b=1)
        elif fam[mid] == "G1":
            kw = ALL_V11[mid][3]
            mc = mc_g1(Ctl11(T0, W0, kw), R0)
            mcb = mc_g1(Ctl11(T0, W0, kw), 6, b=1)
        else:
            kw = ALL_V11[mid][3]
            mc = mc_cb(T0, W0, kw, R0)
            mcb = {"violates": None}
        s = agg(pv[mid], allv)
        b0 = base[fam[mid]]
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
        rows.append({"mutant": mid, "prereg": src[mid], "family": fam[mid], "class": cls[mid],
                     "description": desc[mid], "equiv_class_union": cls_union[mid],
                     "kept_union": cls_union[mid] == mid,
                     "equiv_class_v10": cls_v10.get(mid), "kept_v10": cls_v10.get(mid) == mid,
                     "mc_violates": mc["violates"], "mc_seconds": round(mc["seconds"], 4),
                     "mc_states": mc["n_states"], "mc_witness": (mc.get("witness") or "")[:120],
                     "mc_b1_R6_violates": mcb.get("violates"),
                     "corpus_violating_windows": int(s["viol"]),
                     "corpus_videos_violating": int((pv[mid]["viol"] > 0).sum()),
                     "a": s["a"], "E": s["E"], "miss_all": s["miss_all"],
                     "n_miss_DgeR": int(s["n_miss_DgeR"]), "mAP_proxy": s["mAP_proxy"],
                     "dE_pct": 100 * (s["E"] - b0["E"]) / b0["E"],
                     "dmAP_pts": s["mAP_proxy"] - b0["mAP_proxy"],
                     "dmiss_pts": 100 * (s["miss_all"] - b0["miss_all"]),
                     "monitor_max_delay": max(delays) if delays else None})
    df = pd.DataFrame(rows)
    t_mc = time.time() - t_start

    # ------------------------------------------------------- frontier distance, deltas
    gE = {f: grid_EM(f, np.ones(nv)) for f in ("G1", "CB")}
    dist = {}
    for f in ("G1", "CB"):
        sub = df[df.family == f]
        for mid, (d, dom) in zip(sub.mutant, frontier_distance(sub[["E", "miss_all"]].to_numpy(),
                                                               *gE[f])):
            dist[mid] = (d, dom)
    df["frontier_dist"] = [dist[m][0] for m in df.mutant]
    df["dominated_by_frontier"] = [dist[m][1] for m in df.mutant]
    dtau = {"G1": delta_tau(fg1), "CB": delta_tau(fcb)}
    assert abs(dtau["G1"][0] - DELTA_TAU_V10) < 5e-5, dtau
    dep = {"G1": agg_l(per_video(traces["ORIG"], c, R0, light=True), allv),
           "CB": agg_l(per_video(traces["ORIG_CB"], c, R0, light=True), allv)}
    dep_pv = {"G1": per_video(traces["ORIG"], c, R0, light=True),
              "CB": per_video(traces["ORIG_CB"], c, R0, light=True)}
    dep_dist = {f: frontier_distance(np.array([[dep[f]["E"], dep[f]["miss_all"]]]), *gE[f])[0][0]
                for f in ("G1", "CB")}
    rng_b = np.random.default_rng(a.seed + 100)
    boot_dep = {"G1": [], "CB": []}
    for _ in range(a.B):
        idx = rng_b.integers(0, nv, nv)
        w = np.bincount(idx, minlength=nv).astype(float)
        for f in ("G1", "CB"):
            s = agg_l(dep_pv[f], idx)
            boot_dep[f].append(frontier_distance(np.array([[s["E"], s["miss_all"]]]),
                                                 *grid_EM(f, w))[0][0])
    dboot = {f: float(np.percentile(boot_dep[f], 95)) for f in ("G1", "CB")}
    dt = {f: dtau[f][0] for f in ("G1", "CB")}
    df["delta_tau_fam"] = df.family.map(dt)
    df["delta_boot_fam"] = df.family.map(dboot)
    viol = df.mc_violates == True  # noqa: E712
    df["FI_delta_tau"] = viol & (df.frontier_dist <= df.delta_tau_fam)
    df["FI_half_tau"] = viol & (df.frontier_dist <= df.delta_tau_fam / 2)
    df["FI_double_tau"] = viol & (df.frontier_dist <= 2 * df.delta_tau_fam)
    df["FI_delta_boot"] = viol & (df.frontier_dist <= df.delta_boot_fam)
    df["FI_onesided"] = viol & ((df.frontier_dist <= df.delta_tau_fam) | ~df.dominated_by_frontier)

    # -------------------------------------------------------------- testing oracles (H6c)
    names = list(c["names"])
    order = np.argsort(names)
    pick = order[np.random.default_rng(a.seed).choice(nv, 20, replace=False)]
    rng2 = np.random.default_rng(a.seed + 1)
    subsets = [rng2.choice(nv, 20, replace=False) for _ in range(1000)]

    def caught(mid, idx, oracle):
        s = agg(pv[mid], idx)
        b = agg(pv["ORIG" if fam[mid] == "G1" else "ORIG_CB"], idx)
        if oracle == "A":
            return s["n_miss_DgeR"] > 0
        return (abs(s["mAP_proxy"] - b["mAP_proxy"]) >= 0.5
                or abs(100 * (s["E"] - b["E"]) / b["E"]) >= 1.0)
    df["test20_oracleA"] = [caught(m, pick, "A") for m in df.mutant]
    df["test20_oracleB"] = [caught(m, pick, "B") for m in df.mutant]

    # ------------------------------------------------------------------- bootstrap of FI
    def boot_fi(V, thr_of, seed):
        """v1.0 procedure: rng(seed).integers per sample; frontier + normalisation + mutant
        points recomputed; thresholds fixed. Returns fractions per sample and ship share."""
        rng = np.random.default_rng(seed)
        fr, ship, per = [], [], {m: [] for m in V}
        for _ in range(a.B):
            idx = rng.integers(0, nv, nv)
            w = np.bincount(idx, minlength=nv).astype(float)
            hits = []
            for f in ("G1", "CB"):
                Vf = [m for m in V if fam[m] == f]
                if not Vf:
                    continue
                pts = np.array([[agg(pv[m], idx)["E"], agg(pv[m], idx)["miss_all"]] for m in Vf])
                dd = [d for d, _ in frontier_distance(pts, *grid_EM(f, w))]
                for m, d in zip(Vf, dd):
                    per[m].append(d)
                    hits.append(d <= thr_of[m])
                    if m == SHIP:
                        ship.append(d <= thr_of[m])
            fr.append(float(np.mean(hits)) if hits else float("nan"))
        return fr, ship, per

    def hyp_block(kept_col, label, thr_col_main, n_set):
        kept = df[df[kept_col]]
        Vdf = kept[kept.mc_violates == True]  # noqa: E712
        V = Vdf.mutant.tolist()
        n_v = len(V)
        thr = dict(zip(df.mutant, df[thr_col_main]))
        fr, ship, per = boot_fi(V, thr, a.seed)
        rate_A = {m: float(np.mean([caught(m, s, "A") for s in subsets])) for m in V}
        rate_B = {m: float(np.mean([caught(m, s, "B") for s in subsets])) for m in V}
        corpus_viol = kept[kept.corpus_violating_windows > 0]
        k = int(Vdf.FI_delta_tau.sum())
        out = {
            "label": label, "n_mutants": n_set,
            "n_eff": int(len(kept)), "n_eff_by_family": kept.family.value_counts().to_dict(),
            "n_v": n_v, "violating_kept": V,
            "H4prime": {"k_FI_delta_tau": k, "n_v": n_v, "fraction": k / n_v if n_v else None,
                        "k_FI_half": int(Vdf.FI_half_tau.sum()),
                        "k_FI_double": int(Vdf.FI_double_tau.sum()),
                        "k_FI_onesided": int(Vdf.FI_onesided.sum()),
                        "boot_fraction_mean": float(np.nanmean(fr)) if n_v else None,
                        "boot_fraction_ci95": [float(np.nanpercentile(fr, 2.5)),
                                               float(np.nanpercentile(fr, 97.5))] if n_v else None,
                        "PASS": bool(n_v and k / n_v >= 1 / 3)},
            "H5": {"mutant": SHIP,
                   "frontier_dist": float(df.loc[df.mutant == SHIP, "frontier_dist"].iloc[0]),
                   "boot_share_le_delta": float(np.mean(ship)) if ship else None,
                   "PASS": bool(df.loc[df.mutant == SHIP, "FI_delta_tau"].iloc[0])},
            "H6a": {"mc_catches_all_corpus_violating":
                    bool((corpus_viol.mc_violates == True).all()),  # noqa: E712
                    "n_corpus_violating_kept": int(len(corpus_viol)),
                    "mc_max_seconds": float(kept.mc_seconds.max()),
                    "mc_max_seconds_by_family": kept.groupby("family").mc_seconds.max().to_dict(),
                    "n_mc_violating_not_seen_on_corpus":
                    int(((kept.mc_violates == True) & (kept.corpus_violating_windows == 0)).sum()),  # noqa: E712
                    "n_mc_state_cap": int(kept.mc_violates.isna().sum()),
                    "PASS": bool((corpus_viol.mc_violates == True).all()  # noqa: E712
                                 and kept.mc_seconds.max() < 10)},
            "H6b": {"monitor_max_delay_frames": int(corpus_viol.monitor_max_delay.max())
                    if len(corpus_viol) else None, "R": R0,
                    "PASS": bool(len(corpus_viol) == 0
                                 or corpus_viol.monitor_max_delay.max() <= R0)},
            "H6c": {"videos": [names[i] for i in pick],
                    "caught_oracleA": int(Vdf.test20_oracleA.sum()), "n_v": n_v,
                    "rate_oracleA": float(Vdf.test20_oracleA.mean()) if n_v else None,
                    "caught_oracleB": int(Vdf.test20_oracleB.sum()),
                    "rate_oracleB": float(Vdf.test20_oracleB.mean()) if n_v else None,
                    "PASS": bool(n_v and Vdf.test20_oracleA.mean() < 0.5)},
            "mean_rate_1000_subsets_oracleA": rate_A,
            "mean_rate_1000_subsets_oracleB": rate_B,
            "per_mutant_boot_dist_ci95": {m: [float(np.percentile(v, 2.5)),
                                              float(np.percentile(v, 97.5))]
                                          for m, v in per.items()},
            "per_mutant_dist": {m: float(d) for m, d in zip(Vdf.mutant, Vdf.frontier_dist)},
        }
        return out, Vdf, fr, rate_A

    v10, _, _, _ = hyp_block("kept_v10", "v1.0 set (reproduction of p1_t4.json)",
                             "delta_tau_fam", int((df.prereg == "v1.0").sum()))
    uni, Vu, _, rate_Au = hyp_block("kept_union", "union v1.0 + v1.1 (v1.0 rules)",
                                    "delta_tau_fam", int(len(df)))
    # v1.1 hypotheses on the union
    thr_boot = dict(zip(df.mutant, df.delta_boot_fam))
    frb, shipb, perb = boot_fi(Vu.mutant.tolist(), thr_boot, a.seed)
    n_v = len(Vu)
    k_b = int(Vu.FI_delta_boot.sum())
    k_h = int(Vu.FI_half_tau.sum())
    ship_row = df[df.mutant == SHIP].iloc[0]
    v11 = {
        "delta_boot": dboot, "delta_tau": {f: {"value": dtau[f][0], "n_pairs": dtau[f][1]}
                                           for f in dtau},
        "deployed_correct_point_dist": dep_dist,
        "deployed_correct_boot_dist_median": {f: float(np.median(boot_dep[f])) for f in boot_dep},
        "H4dprime": {"k_FI_delta_boot": k_b, "n_v": n_v, "fraction": k_b / n_v if n_v else None,
                     "by_family": Vu.groupby("family").FI_delta_boot.agg(["sum", "count"]).to_dict(),
                     "boot_fraction_mean": float(np.nanmean(frb)) if n_v else None,
                     "boot_fraction_ci95": [float(np.nanpercentile(frb, 2.5)),
                                            float(np.nanpercentile(frb, 97.5))] if n_v else None,
                     "PASS": bool(n_v and k_b / n_v >= 1 / 3)},
        "H5dprime": {"mutant": SHIP, "frontier_dist": float(ship_row.frontier_dist),
                     "delta_boot_G1": dboot["G1"],
                     "boot_share_le_delta_boot": float(np.mean(shipb)) if shipb else None,
                     "PASS": bool(ship_row.FI_delta_boot)},
        "H6cprime": {"mean_rate_oracleA": rate_Au,
                     "max_rate": float(max(rate_Au.values())) if rate_Au else None,
                     "argmax": max(rate_Au, key=rate_Au.get) if rate_Au else None,
                     "PASS": bool(rate_Au and max(rate_Au.values()) < 0.5)},
        "H7": {"k_FI_half_tau": k_h, "n_v": n_v, "fraction": k_h / n_v if n_v else None,
               "PASS": bool(n_v and k_h / n_v >= 0.5)},
    }
    # reproduction check against p1_t4.json
    p1 = json.loads((RES / "p1_t4.json").read_text(encoding="utf-8"))
    repro = {"n_eff": (v10["n_eff"], p1["n_eff"]), "n_v": (v10["n_v"], p1["n_v_mc"]),
             "k_FI_delta": (v10["H4prime"]["k_FI_delta_tau"], p1["H4prime"]["k_FI_delta"]),
             "boot_ci": (v10["H4prime"]["boot_fraction_ci95"], p1["H4prime"]["boot_fraction_ci95"]),
             "H5_dist": (v10["H5"]["frontier_dist"], p1["H5"]["frontier_dist"]),
             "H6c_caught": (v10["H6c"]["caught_oracleA"], p1["H6c"]["caught_oracleA"]),
             "rate_A": (v10["mean_rate_1000_subsets_oracleA"],
                        p1["H6c"]["mean_rate_1000_subsets_oracleA"])}
    repro_ok = all((abs(x - y) < 1e-9 if isinstance(x, float) else x == y)
                   for k, (x, y) in repro.items() if k not in ("boot_ci", "rate_A")) and \
        np.allclose(repro["boot_ci"][0], repro["boot_ci"][1]) and \
        all(abs(repro["rate_A"][0][m] - repro["rate_A"][1][m]) < 1e-12 for m in repro["rate_A"][1])
    df.to_csv(RES / "mutants_v11.csv", index=False)
    by_class = (df.groupby(["family", "class"])
                .agg(n=("mutant", "size"),
                     eq_orig=("equiv_class_union", lambda s: int(s.str.startswith("EQ_").sum())),
                     kept=("kept_union", "sum"),
                     mc_viol_kept=("mutant", lambda s: int(((df.loc[s.index, "mc_violates"] == True)  # noqa: E712
                                                            & df.loc[s.index, "kept_union"]).sum())))
                .reset_index().to_dict("records"))
    expl = df[df.family == "G1"]
    res = {
        "seed": a.seed, "B": a.B, "hashes": hashes, "n_mutants_union": int(len(df)),
        "n_mutants_v11": int((df.prereg == "v1.1").sum()),
        "n_cb_python_vs_numba_mismatch_frames": n_cb_mismatch,
        "equiv_union": {"n_eq_orig": int((df.equiv_class_union == "EQ_ORIG").sum()),
                        "n_eq_orig_cb": int((df.equiv_class_union == "EQ_ORIG_CB").sum()),
                        "classes": {m: [x for x in df.mutant if cls_union[x] == m]
                                    for m in df.mutant if cls_union[m] == m and
                                    sum(cls_union[x] == m for x in df.mutant) > 1},
                        "eq_orig": [m for m in df.mutant if cls_union[m].startswith("EQ_")]},
        "by_class": by_class,
        "v1.0_on_v1.0_set": v10, "v1.0_reproduces_p1_t4": bool(repro_ok),
        "v1.0_rules_on_union": uni, "v1.1": v11,
        "exploratory_b1_R6": {
            "n_G1": int(len(expl)), "n_violating": int((expl.mc_b1_R6_violates == True).sum()),  # noqa: E712
            "violating": expl[expl.mc_b1_R6_violates == True].mutant.tolist(),  # noqa: E712
            "orig_violates": bool(mc_g1(Ctl(T0, W0, {}), 6, b=1)["violates"])},
        "baselines": base, "deployed_correct": dep,
        "wall_s": {"grid": round(t_grid, 1), "traces": round(t_traces, 1), "mc": round(t_mc, 1),
                   "total": round(time.time() - t_start, 1)},
    }
    (RES / "p2_t4.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    cols = ["mutant", "family", "equiv_class_union", "mc_violates", "mc_seconds",
            "corpus_violating_windows", "dE_pct", "frontier_dist", "FI_delta_tau", "FI_delta_boot",
            "test20_oracleA"]
    print(df[cols].to_string(index=False))
    print(json.dumps({"repro": repro_ok, "v1.1": v11,
                      "union": {k: v for k, v in uni.items() if k.startswith("H")}},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
