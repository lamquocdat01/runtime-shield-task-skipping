"""fi_baserate.py -- base rate of frontier-indistinguishability (FI) and fault mechanisms.

Deterministic, from existing results (no new run of any experiment):
  results/frontier_correct.parquet, results/frontier_correct_cb.parquet  (correct grids, per family)
  results/mutants_v11.csv                                               (87 mutants, union)
  results/p2_t4.json                                                    (delta_tau, delta_boot)
Reports
  - share of the correct configurations of each family whose distance to their own frontier is
    <= delta_boot and <= delta_tau (the FI distance test applied to correct configurations);
  - share of the kept mutants that do NOT violate I_R and pass the same distance test;
  - for every violating kept mutant: distance to the frontier and to the nearest correct
    configuration (normalised plane of the family);
  - the fault mechanism of each violating mutant (from its description in mutants_v11.csv)
    and the number of distinct mechanisms per family.
Usage: python code/fi_baserate.py --seed 42   (seed recorded only)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "code"))
from p1_t4 import frontier_distance  # noqa: E402

RES = REPO / "results"
# mechanism of each violating kept mutant, read from its description (mutants_v11.csv)
MECHANISM = {
    "M1a": "cap fires late (threshold, tick or wrap)", "M5a": "cap fires late (threshold, tick or wrap)",
    "N03": "cap fires late (threshold, tick or wrap)", "N06": "cap fires late (threshold, tick or wrap)",
    "N13": "cap fires late (threshold, tick or wrap)",
    "M5d": "counter advances only on some frames", "N16": "counter advances only on some frames",
    "N18": "counter advances only on some frames",
    "N08": "spurious counter reset", "N32": "spurious counter reset",
    "N21": "stale counter read",
    "C01": "counter reset when BYPASS is left", "C02": "probe does not run the detector",
    "C03": "BYPASS entry frame not dispatched", "C05": "half-rate BYPASS with a bad probe constant",
    "C06": "half-rate BYPASS with a bad probe constant",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    p2 = json.loads((RES / "p2_t4.json").read_text(encoding="utf-8"))["v1.1"]
    grids = {"G1": pd.read_parquet(RES / "frontier_correct.parquet"),
             "CB": pd.read_parquet(RES / "frontier_correct_cb.parquet")}
    mu = pd.read_csv(RES / "mutants_v11.csv")
    kept = mu[mu.kept_union == True]  # noqa: E712
    out = {"seed": a.seed, "families": {}}
    for fam, g in grids.items():
        E, M = g["E"].to_numpy(float), g["miss_all"].to_numpy(float)
        dg = np.array([d for d, _ in frontier_distance(np.column_stack([E, M]), E, M)])
        db, dt = p2["delta_boot"][fam], p2["delta_tau"][fam]["value"]
        k = kept[kept.family == fam]
        nonv = k[k.mc_violates == False]  # noqa: E712
        viol = k[k.mc_violates == True]  # noqa: E712
        e0, e1, m0, m1 = E.min(), E.max(), M.min(), M.max()
        Gn = np.column_stack([(E - e0) / (e1 - e0), (M - m0) / (m1 - m0)])
        rows = []
        for _, r in viol.iterrows():
            p = np.array([(r.E - e0) / (e1 - e0), (r.miss_all - m0) / (m1 - m0)])
            rows.append({"mutant": r.mutant, "mechanism": MECHANISM[r.mutant],
                         "frontier_dist": float(r.frontier_dist),
                         "nearest_correct_dist": float(np.min(np.linalg.norm(Gn - p, axis=1)))})
        mech = sorted({x["mechanism"] for x in rows})
        out["families"][fam] = {
            "n_correct": int(len(g)), "delta_boot": db, "delta_tau": dt,
            "correct_share_le_delta_boot": float(np.mean(dg <= db)),
            "correct_share_le_delta_tau": float(np.mean(dg <= dt)),
            "correct_dist_median": float(np.median(dg)),
            "n_kept_nonviolating": int(len(nonv)),
            "nonviolating_le_delta_boot": int((nonv.frontier_dist <= db).sum()),
            "nonviolating_le_delta_tau": int((nonv.frontier_dist <= dt).sum()),
            "n_violating": int(len(viol)), "n_mechanisms": len(mech), "mechanisms": mech,
            "violating": rows,
            "violating_nearest_correct_max": max(x["nearest_correct_dist"] for x in rows),
        }
    (RES / "fi_baserate.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for fam, f in out["families"].items():
        print(fam, {k: v for k, v in f.items() if k not in ("violating", "mechanisms")})


if __name__ == "__main__":
    main()
