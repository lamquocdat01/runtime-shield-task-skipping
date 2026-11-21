"""reproduce.py -- runs the whole pipeline in dependency order and regenerates
results/numbers.json; with --check, compares it key by key with a reference numbers.json
(writes REPRO_CHECK.md). Python-only steps always run; steps needing the WSL toolchain
(IMITATOR 3.4, GHC + Copilot 4.8.1 + copilot-verifier, gcc) run with --tools.
Inputs: THS_DATASETS (default ./datasets) must contain the derived corpus described in
DATA.md (gate scores + always-on detections); no frames or ground truth are needed.
Usage: python code/reproduce.py --seed 42 [--tools] [--skip-region] [--check ref.json]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(args: list[str], env=None) -> float:
    t0 = time.time()
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT, env=env)
    return time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tools", action="store_true", help="also run the WSL tool steps")
    ap.add_argument("--skip-region", action="store_true",
                    help="skip the 642,600-point region grid (~5 min, 4 processes)")
    ap.add_argument("--seeds5", action="store_true",
                    help="also repeat the stochastic steps with five seeds (~20 min, code/seeds5.py)")
    ap.add_argument("--check", default="")
    a = ap.parse_args()
    s = ["--seed", str(a.seed)]
    steps = [["code/frontier_sweep.py", *s], ["code/p1_shield_check.py", *s],
             ["code/conformance.py", *s], ["code/p1_t4.py", *s]]
    if not a.skip_region:
        steps.append(["code/region_bruteforce.py", *s])
    steps += [["code/cb_exact_run.py", *s], ["code/cb_rule_ext.py", *s], ["code/p2_t4.py", *s],
              ["code/fi_baserate.py", *s]]
    if a.seeds5:
        steps.append(["code/seeds5.py", "--run"])
    if a.tools:
        steps += [["code/imitator_cb_final.py", *s], ["code/shield_bridge.py", *s]]
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    times = {}
    if a.tools:     # Copilot: regenerate C, prove properties, run copilot-verifier
        t0 = time.time()
        sh = str(ROOT / "monitor" / "build_shield.sh").replace("\\", "/")
        sh = "/mnt/" + sh[0].lower() + sh[2:]
        subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "--", "bash", sh],
                       check=True, env=env, capture_output=True)
        times["monitor/build_shield.sh"] = round(time.time() - t0, 1)
        sz = sh.replace("build_shield.sh", "size_shield.sh")
        subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "--", "bash", sz], check=True, env=env,
                       capture_output=True)
    for st in steps:
        times[st[0]] = round(run(st, env), 1)
    chk = ["code/make_numbers.py"] + (["--check", a.check] if a.check else [])
    run(chk, env)
    print(times)


if __name__ == "__main__":
    main()
