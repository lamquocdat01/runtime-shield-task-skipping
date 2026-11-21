"""p0_crosscheck_deep3.py -- compare the independent re-implementation (policy_ref.py)
against the per-event outcomes of a re-run of the released Deep3 harness after
correcting a sequence-ordering error in its LASIESTA corpus listing (frames in
capture order; re-run 27-09-2026; events_gated.csv = deployed `>`,
events_gated_fixed.csv = `>=`). These corrected tables differ from the tables
printed in the Deep3 article.
Read-only on the Deep3 side; nothing is imported from it.

Usage: python code/p0_crosscheck_deep3.py --seed 42 --deep3-dir <dir with events_gated*.csv>
Appends a "crosscheck_deep3" block to results/p0_trace_stats.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

TOPIC_DIR = Path(__file__).resolve().parent.parent
OUT_ROOT = Path(os.environ.get("THS_DATASETS", "datasets")) / "_derived" / "06"
# processed per-event tables of the corrected Deep3 re-run (not redistributed); set DEEP3_PROCESSED
DEFAULT_DEEP3 = os.environ.get("DEEP3_PROCESSED", "deep3_processed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deep3-dir", default=DEFAULT_DEEP3)
    ap.add_argument("--stats", default=str(TOPIC_DIR / "results" / "p0_trace_stats.json"))
    a = ap.parse_args()
    key = ["dataset", "video", "onset", "duration"]
    out = {"seed": a.seed, "deep3_dir": a.deep3_dir}
    for var, fn in (("gt", "events_gated.csv"), ("ge", "events_gated_fixed.csv")):
        ref = pd.read_csv(Path(a.deep3_dir) / fn)
        ref["video"] = ref["video"].astype(str)
        mine = pd.read_parquet(OUT_ROOT / f"events_{var}.parquet")
        m = mine.merge(ref, on=key, how="outer", suffixes=("", "_d3"), indicator=True)
        both = m[m["_merge"] == "both"]
        out[var] = {
            "n_ref": int(len(ref)), "n_mine": int(len(mine)),
            "n_matched_events": int(len(both)),
            "n_only_mine": int((m["_merge"] == "left_only").sum()),
            "n_only_ref": int((m["_merge"] == "right_only").sum()),
            "missed_agree": int((both.missed == both.missed_d3).sum()),
            "caught_by_gate_agree": int((both.caught_by_gate == both.caught_by_gate_d3).sum()),
            "latency_agree": int(((both.latency == both.latency_d3.fillna(-1))).sum()),
            "missed_mine": int(mine.missed.sum()), "missed_ref": int(ref.missed.sum()),
        }
        print(var, out[var])
    st = json.loads(Path(a.stats).read_text(encoding="utf-8"))
    st["crosscheck_deep3"] = out
    Path(a.stats).write_text(json.dumps(st, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
