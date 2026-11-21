"""conformance.py -- P1 Viec 4 (item 13): trace conformance of the regenerated deployment
traces against the timed-automaton model (validity of T1, Lemma 1).

The TA of model/toy_refresh.xml (one frame per time unit, gate = nondeterministic select g)
is re-written here as an explicit transition function on the discrete state
(ph = t mod T, first, fsl, gap); clock x only enforces one decision per unit, so runs of the TA
that let time progress are exactly the sequences of `step` below (Lemma 1, step 1).
Two checks per video and per variant ('>' deployed, '>=' fixed), n = 121 videos:
  (i)  input/output conformance: with the LOGGED gate request g_t, the TA output equals
       the logged o_t at every frame;
  (ii) trace inclusion: the logged OPEN/CLOSED sequence o is a run of the TA for SOME gate
       sequence (subset construction over TA states).
Discrimination check: each variant's traces are also replayed on the OTHER variant's TA;
(i) must then fail on the videos where the two variants differ.
Inputs: _derived/06/traces_{gt,ge}.parquet.  Output: results/p1_conformance.json
Usage:  python code/conformance.py --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_ref import OUT_ROOT  # noqa: E402

TOPIC_DIR = Path(__file__).resolve().parent.parent
CAP = 64


def ta_step(s, g: int, T: int, W: int, ge: bool):
    ph, first, fsl, gap = s
    rf = (not first) and ph == 0
    cap = fsl >= W if ge else fsl > W
    o = g == 1 or rf or cap
    if o:
        fsl, gap = 0, 0
    else:
        fsl, gap = min(fsl + 1, CAP), min(gap + 1, CAP)
    return o, ((ph + 1) % T, False, fsl, gap)


def ta_init(W: int):
    return (0, True, W + 1, 0)


def io_conform(g, o, T, W, ge) -> int:
    """Return the first frame index where the TA output differs, or -1."""
    s = ta_init(W)
    for t, (gt, ot) in enumerate(zip(g, o)):
        y, s = ta_step(s, int(gt), T, W, ge)
        if y != bool(ot):
            return t
    return -1


def included(o, T, W, ge) -> bool:
    S = {ta_init(W)}
    for ot in o:
        nxt = set()
        for s in S:
            for g in (0, 1):
                y, ns = ta_step(s, g, T, W, ge)
                if y == bool(ot):
                    nxt.add(ns)
        if not nxt:
            return False
        S = nxt
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="unused (deterministic); recorded")
    ap.add_argument("--T", type=int, default=15)
    ap.add_argument("--W", type=int, default=4)
    a = ap.parse_args()
    out = {"seed": a.seed, "T": a.T, "W": a.W, "model": "model/toy_refresh.xml (Python TA)"}
    for var, ge in (("gt", False), ("ge", True)):
        fr = pd.read_parquet(OUT_ROOT / f"traces_{var}.parquet")
        n = io_ok = inc_ok = cross_fail = frames = 0
        first_bad = []
        for (ds, v), x in fr.groupby(["dataset", "video"], sort=True):
            g, o = x.g_t.to_numpy(), x.o_t.to_numpy()
            n += 1
            frames += len(o)
            k = io_conform(g, o, a.T, a.W, ge)
            io_ok += k < 0
            if k >= 0 and len(first_bad) < 5:
                first_bad.append(f"{ds}/{v}@{k}")
            inc_ok += included(o, a.T, a.W, ge)
            cross_fail += io_conform(g, o, a.T, a.W, not ge) >= 0
        out[var] = {"n_videos": n, "n_frames": frames,
                    "io_conformant_videos": io_ok, "trace_included_videos": inc_ok,
                    "io_rate": io_ok / n, "inclusion_rate": inc_ok / n,
                    "first_nonconformant": first_bad,
                    "replayed_on_other_variant_io_fail_videos": cross_fail}
        print(var, out[var])
    (TOPIC_DIR / "results" / "p1_conformance.json").write_text(json.dumps(out, indent=1),
                                                              encoding="utf-8")


if __name__ == "__main__":
    main()
