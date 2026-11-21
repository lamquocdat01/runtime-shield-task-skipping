"""shield_bridge.py -- P2 Viec 3 (items 8-9): run the copilot-verifier-proved C (monitor/
gate_shield.c, gate_ctl.c) on the 121-video corpus through ctypes, and time it.

Windows side (default):  prepares the inputs (gate request at tau = 0.55, the shipped '>'
  trace, the fixed '>=' trace, video-start flags) -> $THS_DATASETS/_derived/06/shield_inputs.npz
  and raw uint8 files for the benchmark, then calls WSL (no C compiler on the Windows host) to
  build libshield.so / libctl.so / bench_shield with gcc -O2 and to run this script with --run,
  then runs the benchmark (>= 10 runs x 166,589 frames) and writes results/p2_shield.json.
WSL side (--run): numpy + ctypes only.
  (a) open loop: L(pi) with pi = shipped '>' trace; extra OPEN fraction (P1: 3.16 %), and
      frame-by-frame equality with the Python reference lazy() of code/p1_shield_check.py;
  (b) closed loop (Prop. 4.4): the shipped controller G1_> whose fsl is reset by the C
      shield's OPENs, one ctypes call per frame -> must equal the fixed '>=' trace on all
      166,589 frames;
  (c) the verified C controller (three variants) vs the Python reference traces.
Overhead: median ns/frame over the runs, mean with expanded uncertainty U = 2 s / sqrt(n)
(coverage factor k = 2), compared with the detector cost from Deep3 Table III
(DOI 10.1109/JIOT.2026.3740931): fastest stack YOLO26s TRT fp16 on Orin Nano, 26.7 fps.
Usage: python code/shield_bridge.py --seed 42 [--runs 30]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOPIC_DIR = HERE.parent
R0, T0, W0 = 5, 15, 4
DERIVED = Path(os.environ.get("THS_DATASETS", "datasets")) / "_derived" / "06"
# Deep3 Table III (fps per platform/detector/runtime); ms/frame = 1000 / fps
DETECTORS = {"Orin Nano YOLO26s TRT fp16": 26.7, "Orin Nano YOLOv3u TRT fp16": 20.5,
             "Orin Nano YOLO26s ORT CUDA": 17.6, "Orin Nano YOLOv3u ORT CUDA": 7.9,
             "embedded GPU YOLOv3 (orig.)": 30.0}


def wsl_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:]


def lazy_ref(o: np.ndarray, R: int) -> np.ndarray:
    """= code/p1_shield_check.lazy (copied: the WSL side has no access to numba/pandas)."""
    out = o.copy()
    c = 0
    for t in range(len(out)):
        if not out[t] and c >= R - 1:
            out[t] = True
        c = 0 if out[t] else c + 1
    return out


# ------------------------------------------------------------------------------ WSL side
def run_wsl(inp: Path, lib_dir: Path, out_json: Path) -> None:
    import ctypes
    z = np.load(inp)
    pi, valid, rst, g = (z[k].astype(np.uint8) for k in ("trace_gt", "valid", "rst", "g"))
    ge, gt, vs = z["trace_ge"].astype(bool), z["trace_gt"].astype(bool), z["vstart"]
    n = len(pi)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    sh = ctypes.CDLL(str(lib_dir / "libshield.so"))
    sh.shield_batch.restype = ctypes.c_long
    sh.shield_frame.restype = ctypes.c_int
    ct = ctypes.CDLL(str(lib_dir / "libctl.so"))
    res = {"n_frames": int(n), "n_videos": int(len(vs) - 1)}

    def ptr(a):
        return a.ctypes.data_as(u8p)
    # (a) open loop
    e = np.zeros(n, dtype=np.uint8)
    sh.shield_batch(ptr(pi), ptr(valid), ptr(rst), ptr(e), ctypes.c_long(n))
    e = e.astype(bool)
    ref = np.zeros(n, dtype=bool)
    for v in range(len(vs) - 1):
        ref[vs[v]:vs[v + 1]] = lazy_ref(gt[vs[v]:vs[v + 1]], R0)

    def max_run(o):
        m = 0
        for v in range(len(vs) - 1):
            r = 0
            for x in o[vs[v]:vs[v + 1]]:
                r = 0 if x else r + 1
                m = max(m, r)
        return m
    res["open_loop"] = {"extra_open_fraction": float((e.sum() - gt.sum()) / n),
                        "n_extra_open": int(e.sum() - gt.sum()),
                        "only_add": bool(np.all(e >= gt)),
                        "frames_equal_python_lazy": int((e == ref).sum()),
                        "max_closed_run": max_run(e)}
    # (b) closed loop: G1_> reads the shield output (fsl reset by e)
    f = ctypes.c_int
    ec = np.zeros(n, dtype=bool)
    t0 = time.perf_counter()
    for v in range(len(vs) - 1):
        fsl = W0 + 1
        for t in range(vs[v], vs[v + 1]):
            pos = t - vs[v]
            if valid[t]:
                p = bool(g[t]) or (pos > 0 and pos % T0 == 0) or fsl > W0
            else:
                p = False
            r = sh.shield_frame(f(int(p)), f(int(valid[t])), f(int(pos == 0)))
            ec[t] = bool(r & 1)
            fsl = 0 if ec[t] else fsl + 1
    res["closed_loop"] = {"frames_equal_fixed_ge": int((ec == ge).sum()),
                          "extra_open_fraction_vs_shipped": float((ec.sum() - gt.sum()) / n),
                          "max_closed_run": max_run(ec), "wall_s": round(time.perf_counter() - t0, 1)}
    # (c) verified C controller, three variants
    oc = np.zeros(n, dtype=np.uint8)
    ct.ctl_batch(ptr(g), ptr(valid), ptr(rst), ptr(oc), ctypes.c_long(n))
    res["c_controller"] = {"o_ge_equal_python_ge": int(((oc & 1) > 0).__eq__(ge).sum()),
                           "o_gt_equal_python_gt": int(((oc & 2) > 0).__eq__(gt).sum()),
                           "o_gtL_equal_python_ge": int(((oc & 4) > 0).__eq__(ge).sum())}
    out_json.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


# -------------------------------------------------------------------------- Windows side
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="unused (deterministic); recorded")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--run", action="store_true", help="WSL side")
    ap.add_argument("--inp", default="")
    ap.add_argument("--lib", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.run:
        run_wsl(Path(a.inp), Path(a.lib), Path(a.out))
        return
    sys.path.insert(0, str(HERE))
    from frontier_sweep import build_corpus, controller, gate_request
    c = build_corpus()
    vs = c["vstart"]
    g = gate_request(c["score"], vs, 0.55, 0.30, 12, 6)
    gt = controller(g, c["valid"], vs, T0, W0, False)
    ge = controller(g, c["valid"], vs, T0, W0, True)
    rst = np.zeros(len(g), dtype=np.uint8)
    rst[vs[:-1]] = 1
    DERIVED.mkdir(parents=True, exist_ok=True)
    inp = DERIVED / "shield_inputs.npz"
    np.savez_compressed(inp, g=g, valid=c["valid"], rst=rst, trace_gt=gt, trace_ge=ge, vstart=vs)
    bench = DERIVED / "shield_bench"
    bench.mkdir(exist_ok=True)
    for nm, arr in (("pi", gt), ("valid", c["valid"]), ("rst", rst)):
        arr.astype(np.uint8).tofile(bench / f"{nm}.u8")
    mon = TOPIC_DIR / "monitor"
    wl = "/opt/vgc/shield/bridge"
    script = f"""set -e
mkdir -p {wl} && cd {wl}
cp '{wsl_path(mon)}'/gate_shield.[ch] '{wsl_path(mon)}'/gate_ctl.[ch] '{wsl_path(mon)}'/shield_glue.c '{wsl_path(mon)}'/bench_shield.c .
touch gate_shield_types.h gate_ctl_types.h
gcc --version | head -1
gcc -O2 -std=c99 -Wall -fPIC -shared -Wl,-Bsymbolic -DSHIELD gate_shield.c shield_glue.c -o libshield.so
gcc -O2 -std=c99 -Wall -fPIC -shared -Wl,-Bsymbolic -DCTL gate_ctl.c shield_glue.c -o libctl.so
gcc -O2 -std=c99 -Wall -DSHIELD gate_shield.c shield_glue.c bench_shield.c -o bench_shield
python3 '{wsl_path(HERE / "shield_bridge.py")}' --run --inp '{wsl_path(inp)}' --lib {wl} --out '{wsl_path(TOPIC_DIR / "results" / "p2_shield_wsl.json")}'
lscpu | grep -E 'Model name|^CPU\\(s\\)' ; uname -r
taskset -c 2 ./bench_shield '{wsl_path(bench)}/pi.u8' '{wsl_path(bench)}/valid.u8' '{wsl_path(bench)}/rst.u8' {a.runs} > bench.txt
cat bench.txt
"""
    sp = TOPIC_DIR / "results" / "shield_bridge_wsl.sh"
    sp.write_text(script, encoding="utf-8", newline="\n")
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    r = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "--", "bash", wsl_path(sp)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    log = TOPIC_DIR / "results" / "shield_bridge.log"
    log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads((TOPIC_DIR / "results" / "p2_shield_wsl.json").read_text(encoding="utf-8"))
    rows = [ln.split() for ln in r.stdout.splitlines() if len(ln.split()) == 4
            and ln.split()[0].isdigit()]
    nsf = np.array([float(x[2]) for x in rows])
    k = len(nsf)
    mean, sd = float(nsf.mean()), float(nsf.std(ddof=1))
    U = 2 * sd / np.sqrt(k)
    res["overhead"] = {"runs": k, "frames_per_run": res["n_frames"],
                       "ns_per_frame_median": float(np.median(nsf)),
                       "ns_per_frame_mean": mean, "ns_per_frame_sd": sd,
                       "U_k2_of_mean": float(U), "ns_per_frame_min": float(nsf.min()),
                       "ns_per_frame_max": float(nsf.max()),
                       "host": "i7-1185G7, WSL2 Ubuntu 24.04, gcc 13.3 -O2, pinned to CPU 2",
                       "detector_ms_per_frame": {d: 1000 / fps for d, fps in DETECTORS.items()},
                       "ratio_shield_to_detector": {d: float(np.median(nsf)) * 1e-9 / (1 / fps)
                                                    for d, fps in DETECTORS.items()},
                       "detector_source": "Deep3 Table III, DOI 10.1109/JIOT.2026.3740931"}
    res["p1_reference"] = {"open_loop_extra_open_fraction": 0.031611}
    res["seed"] = a.seed
    (TOPIC_DIR / "results" / "p2_shield.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
