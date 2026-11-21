"""cb_exact_run.py -- regenerates the two P1 circuit-breaker files that were produced by direct
calls of region_bruteforce.cb_exact_main / lmax_cb_exact on 16-10-2026 (no script existed):
  results/region_ext/cb_exact.json          24 points with the real constants K = 300, P = 30
  results/region_ext/cb_exact_KP_rule.json  the d | gcd(K, P) rule on 816 cases (K <= 15)
Both use lmax_cb_exact (refresh phase pinned to pos % T == 0, as in P1); the extended,
phase-adversarial check of Proposition 2.7 is code/cb_rule_ext.py.
Usage: python code/cb_exact_run.py --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from region_bruteforce import OUT_DIR, cb_exact_main, lmax_cb_exact, rstar  # noqa: E402

POINTS = ([(15, 4, ge, d, 1 if d >= 2 else 0) for ge in (1, 0) for d in range(6)]
          + [(T, W, 1, 3, 1) for T in (5, 10, 20, 30) for W in (2, 8, 15)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="unused (exhaustive); recorded")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = cb_exact_main(POINTS, K=300, P=30)
    (OUT_DIR / "cb_exact.json").write_text(json.dumps({"seed": a.seed, "K": 300, "P": 30,
                                                       "rows": rows}, indent=1), encoding="utf-8")
    n = mism = 0
    head = []
    for (T, W) in ((15, 4), (30, 8)):
        for d in range(2, 6):
            g1 = rstar(T, W, 1, d, 1, 0, Fraction(1), 0)[1]
            for K in range(4, 16):
                for P in range(1, K):
                    hole = int(lmax_cb_exact(T, W, 1, d, 1, K, P)) + d + 1 > g1
                    rule = (K - P) % d != 0 or K % d != 0
                    n += 1
                    if hole != rule:
                        mism += 1
                        if len(head) < 10:
                            head.append([T, W, d, K, P, hole])
    (OUT_DIR / "cb_exact_KP_rule.json").write_text(json.dumps(
        {"seed": a.seed, "grid": "K 4..15, P 1..K-1, d 2..5 (block), (T,W) in {(15,4),(30,8)}, ge",
         "rule_tested": "hole iff (K-P) mod d != 0 or K mod d != 0", "n": n,
         "n_rule_mismatch": mism, "rule_mismatch_head": head}, indent=1), encoding="utf-8")
    print(n, mism)


if __name__ == "__main__":
    main()
