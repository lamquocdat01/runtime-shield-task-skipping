"""seeds5.py -- repeat every stochastic step of the analysis with five explicit seeds.

The stochastic quantities of the paper are the video-level bootstrap intervals (B = 1000), the
random-valid enforcer baseline, the bootstrap threshold delta_boot and the 1000 random 20-video
subsets. They come from three scripts, each with --seed:
  code/p1_shield_check.py   Table I (enforcers, bootstrap intervals, random-valid baseline)
  code/p1_t4.py             v1.0 study (bootstrap intervals, random subsets)
  code/p2_t4.py             v1.0 on the union and v1.1 (delta_boot, bootstrap intervals, subsets)
Each seed s runs the three scripts in a fresh copy of the repository (so results/ of the
repository is not overwritten) and keeps their JSON outputs in results/seeds5/seed_<s>/.

Seeds: 42 (the seed of the pre-registrations) and 1001, 2002, 3003, 4004. The scripts derive
streams as seed, seed + 1 and seed + 100, so spaced seeds keep every stream distinct.
Not part of this sensitivity analysis: the registered 20-video subset of H6c, which the
pre-registration fixes as default_rng(42).choice(...) independently of --seed; for s != 42 the
scripts draw a different subset, which is therefore not reported.

Usage:
  python code/seeds5.py --run [--seeds 42 1001 2002 3003 4004] [--work <scratch dir>]
  python code/seeds5.py --summarize            # -> results/seeds5_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "seeds5"
SEEDS = [42, 1001, 2002, 3003, 4004]
SCRIPTS = ["p1_shield_check.py", "p1_t4.py", "p2_t4.py"]
KEEP = ["p1_shield.json", "p1_t4.json", "p2_t4.json"]


def run(seeds: list[int], work: Path) -> None:
    for s in seeds:
        dst = work / f"seed_{s}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(REPO, dst, ignore=shutil.ignore_patterns(".git", ".venv", "seeds5", "__pycache__"))
        # deterministic input of p2_t4.py (frontier grid); its --seed is only recorded
        subprocess.run([sys.executable, "code/frontier_sweep.py", "--seed", "42"], cwd=dst, check=True,
                       env=dict(os.environ))
        for sc in SCRIPTS:
            print(f"seed {s}: {sc}", flush=True)
            subprocess.run([sys.executable, f"code/{sc}", "--seed", str(s)], cwd=dst, check=True,
                           env=dict(os.environ))
        o = OUT / f"seed_{s}"
        o.mkdir(parents=True, exist_ok=True)
        for f in KEEP:
            shutil.copy2(dst / "results" / f, o / f)
        shutil.rmtree(dst)


def metrics(d: Path) -> dict:
    sh = json.loads((d / "p1_shield.json").read_text(encoding="utf-8"))["c_corpus"]["enforcers"]
    p2 = json.loads((d / "p2_t4.json").read_text(encoding="utf-8"))
    m: dict = {}
    for e in ["lazy", "periodic", "eager", "random", "lazy_closed_loop"]:
        x = sh[e]
        m[f"enf.{e}.extra_pct"] = 100 * x["extra_activation"]
        m[f"enf.{e}.ci_lo_pct"] = 100 * x["ci95"][0]
        m[f"enf.{e}.ci_hi_pct"] = 100 * x["ci95"][1]
        m[f"enf.{e}.diff_pts"] = 100 * x["diff_vs_lazy"]
        m[f"enf.{e}.diff_lo_pts"] = 100 * x["diff_vs_lazy_ci95"][0]
        m[f"enf.{e}.diff_hi_pts"] = 100 * x["diff_vs_lazy_ci95"][1]
        m[f"enf.{e}.videos_fewer_than_lazy"] = x["n_videos_where_fewer_opens_than_lazy"]
    v10, un, v11 = p2["v1.0_on_v1.0_set"], p2["v1.0_rules_on_union"], p2["v1.1"]
    m["h4p_v10.ci_lo"], m["h4p_v10.ci_hi"] = v10["H4prime"]["boot_fraction_ci95"]
    m["h4p_v10.PASS"] = v10["H4prime"]["PASS"]
    m["h5_v10.boot_share"] = v10["H5"]["boot_share_le_delta"]
    m["h4p_union.ci_lo"], m["h4p_union.ci_hi"] = un["H4prime"]["boot_fraction_ci95"]
    m["h4p_union.PASS"] = un["H4prime"]["PASS"]
    m["delta_boot.G1"] = v11["delta_boot"]["G1"]
    m["delta_boot.CB"] = v11["delta_boot"]["CB"]
    m["h4dp.k"] = v11["H4dprime"]["k_FI_delta_boot"]
    m["h4dp.k_G1"] = v11["H4dprime"]["by_family"]["sum"]["G1"]
    m["h4dp.k_CB"] = v11["H4dprime"]["by_family"]["sum"]["CB"]
    m["h4dp.ci_lo"], m["h4dp.ci_hi"] = v11["H4dprime"]["boot_fraction_ci95"]
    m["h4dp.PASS"] = v11["H4dprime"]["PASS"]
    m["h5dp.boot_share"] = v11["H5dprime"]["boot_share_le_delta_boot"]
    m["h5dp.PASS"] = v11["H5dprime"]["PASS"]
    m["h6cp.max_rate"] = v11["H6cprime"]["max_rate"]
    m["h6cp.argmax"] = v11["H6cprime"]["argmax"]
    m["h6cp.PASS"] = v11["H6cprime"]["PASS"]
    m["h7.PASS"] = v11["H7"]["PASS"]
    ra, rb = un["mean_rate_1000_subsets_oracleA"], un["mean_rate_1000_subsets_oracleB"]
    m["subsets.max_rate_G1"] = max(v for k, v in ra.items() if not k.startswith("C"))
    m["subsets.n_zero_rate_A"] = sum(v == 0 for v in ra.values())
    m["subsets.n_invisible"] = sum(ra[k] == 0 and rb[k] == 0 for k in ra)
    return m


def summarize() -> None:
    seeds = sorted(int(p.name.split("_")[1]) for p in OUT.glob("seed_*"))
    per = {s: metrics(OUT / f"seed_{s}") for s in seeds}
    keys = list(per[seeds[0]])
    summ = {}
    for k in keys:
        vals = [per[s][k] for s in seeds]
        if all(isinstance(v, bool) for v in vals) or all(isinstance(v, str) for v in vals):
            summ[k] = {"values": vals, "all_equal": len(set(vals)) == 1}
        else:
            summ[k] = {"values": vals, "mean": statistics.fmean(vals),
                       "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                       "min": min(vals), "max": max(vals)}
    verdict_keys = [k for k in keys if k.endswith(".PASS")]
    flips = [k for k in verdict_keys if not summ[k]["all_equal"]]
    res = {"seeds": seeds, "n_seeds": len(seeds), "per_seed": {str(s): per[s] for s in seeds},
           "summary": summ, "verdicts": verdict_keys, "verdict_flips": flips,
           "n_verdict_flips": len(flips),
           "note": "registered H6c subset is fixed by the pre-registration (default_rng(42)) and not varied"}
    (REPO / "results" / "seeds5_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"seeds {seeds}; verdict flips: {flips}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--work", default="")
    a = ap.parse_args()
    if a.run:
        work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="seeds5_"))
        work.mkdir(parents=True, exist_ok=True)
        run(a.seeds, work)
    if a.summarize or a.run:
        summarize()


if __name__ == "__main__":
    main()
