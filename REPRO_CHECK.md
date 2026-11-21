# REPRO_CHECK — numbers.json, key by key

Reference: `paper/numbers.json` (the numbers printed in the manuscript) · regenerated: `results/numbers.json` of a fresh copy of this repository in a new Python 3.11 virtual environment (`pip install -r requirements.txt`), Windows 11 + WSL2 Ubuntu 24.04, seed 42, 21 November 2026.

- stable keys in reference: **2181**, of which compared: **2180**; identical (rel. tol. 1e-9): **2180**; different: **0**; missing in the rerun: **0**; new keys: 40
- volatile (timing) keys, not compared: 145
- stable keys compared as volatile (sole exception, see below): 1

| Source file | keys | identical | missing |
|---|---|---|---|
| bruteforce_summary.json | 46 | 46 | 0 |
| cb_exact.json | 243 | 243 | 0 |
| cb_exact_KP_rule.json | 3 | 3 | 0 |
| cb_rule_ext.json | 127 | 127 | 0 |
| fi_baserate.json | 57 | 57 | 0 |
| frontier_correct_info.json | 24 | 24 | 0 |
| gate_shield_verify.log | 14 | 14 | 0 |
| imitator_cb_final.json | 36 | 36 | 0 |
| imitator_summary.json | 373 | 373 | 0 |
| p1_conformance.json | 17 | 17 | 0 |
| p1_shield.json | 71 | 71 | 0 |
| p1_t4.json | 53 | 53 | 0 |
| p2_shield.json | 22 | 22 | 0 |
| p2_t4.json | 284 | 284 | 0 |
| seeds5_summary.json | 802 | 802 | 0 |
| shield_size.json | 8 | 8 | 0 |

## Sole exception: exit code of a run without result

| key | reference | rerun |
|---|---|---|
| `imitator_summary.json:runs.CB_d_block_k1-1_gt.returncode` | 1 | 124 |

The six IMITATOR breaker runs of the template encoding (`CB_k1-1_ge/gt`, `CB_d_pipe_k1-1_ge/gt`, `CB_d_block_k1-1_ge/gt`, from `code/region_imitator.py`) produce no result in either run (`integer_check.n_points` = 0) and are not used by the paper; its breaker checks use the discrete encoding of `code/imitator_cb_final.py`. Five exit with code 124 (900 s timeout) in both runs. `CB_d_block_k1-1_gt` exited with code 1 after 407 s in the reference run (cause not determined) and with 124 at the timeout in the reproduction. Its `returncode` is therefore compared like the volatile `wall_s` keys; this is the only stable key so treated.


## How the reproduction was run

```
python code/reproduce.py --seed 42 --tools --seeds5 --check paper/numbers.json
python code/region_imitator.py --seed 42
python code/make_numbers.py --check paper/numbers.json
```

## New keys

The 40 new keys are the per-run `box.*` entries (`TMAX`, `WMAX`, `RMAX`, `DMAX`, `BMAX`) of eight IMITATOR runs without the breaker (`b`, `d_b_pipe`, `d_block`, `d_pipe`, both comparisons), which the reference summary did not record per run. No manuscript number is taken from them; every key of the reference is present and compared.
