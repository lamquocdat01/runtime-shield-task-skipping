"""p0_sanity.py -- P0 counterexample hunt for the draft theory (theory/THEORY_06_draft.md).

(a) Parametric certificate region. Exhaustive explicit-state exploration of the toy
    controller (adversarial gate) over an integer grid of (T, W): the maximal reachable
    OPEN-to-OPEN gap must equal min(T, W+1) under '>=' and min(T, W+2) under '>'.
    If IMITATOR is available in WSL (Ubuntu-24.04), the parametric model
    model/toy_refresh_param.imi is also run (EF-synthesis) and its constraint is
    evaluated on the same integer grid.
(b) Monitor I_R on the regenerated 121-video traces (deployed '>' and fixed '>='):
    a saturating counter of consecutive CLOSED frames (ceil(log2 R) bits) must flag
    every window of R consecutive CLOSED frames, with delay <= R frames from the first
    frame of the window, and raise no alarm on '>=' traces. A shield variant (force
    OPEN when the counter would reach R) must restore R-refresh at an extra activation
    <= 1/R. Plus a randomized differential test against a brute-force window check.
(c) Mutant classes: 6 classes x 6 mutants = 36 listed; a quick run of 10 mutants on
    20 videos (seed-chosen) reports which are benchmark-invisible:
    |dmAP-proxy| < 0.5 points, |dE| < 1 %, and I_R violated.

Usage: python code/p0_sanity.py --seed 42
Output: results/p0_sanity.json (+ results/p0_mutants_quick.csv)
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_ref import (CORPUS, DET_ROOT, GATE_TRACE, OUT_ROOT, extract_events,  # noqa: E402
                        hysteresis, load_detections)

TOPIC_DIR = Path(__file__).resolve().parent.parent
E_IDLE, E_ACTIVE = 30.4, 290.8   # mJ/frame, Deep3 [D3-paper] l.141


# ---------------------------------------------------------------- (a) region
def max_gap_explicit(T: int, W: int, ge: bool) -> tuple[int, list[int]]:
    """BFS over (ph, first, fsl) with adversarial gate; returns (max gap, witness)."""
    start = (0, True, W + 1, 0)            # ph, first, fsl, closed-run
    seen = {start: None}
    frontier = [start]
    best, best_state = 0, start
    while frontier:
        nxt = []
        for s in frontier:
            ph, first, fsl, run = s
            for g in (0, 1):
                rf = (not first) and ph == 0
                cap = (fsl >= W) if ge else (fsl > W)
                o = g == 1 or rf or cap
                ns = ((ph + 1) % T, False, 0 if o else fsl + 1, 0 if o else run + 1)
                if ns not in seen:
                    seen[ns] = (s, int(o))
                    nxt.append(ns)
                    if ns[3] > best:
                        best, best_state = ns[3], ns
        frontier = nxt
    # reconstruct OPEN/CLOSED witness ending in the longest closed run
    trace, s = [], best_state
    while seen[s] is not None:
        prev, o = seen[s]
        trace.append(o)
        s = prev
    return best + 1, trace[::-1]           # OPEN-to-OPEN gap = closed run + 1


def run_imitator(tmax: int, wmax: int, rmax: int, ge: bool, workdir: Path) -> dict:
    """Run IMITATOR (WSL) on the parametric toy; return constraint text or error."""
    src = (TOPIC_DIR / "model" / "toy_refresh_param.imi").read_text(encoding="utf-8")
    capoff = 1 if ge else 2
    model = (src.replace("W + CAPOFF - 1", f"W + {capoff - 1}")
                .replace("W + CAPOFF", f"W + {capoff}")
                .replace("TMAX", str(tmax)).replace("WMAX", str(wmax))
                .replace("RMAX", str(rmax)))
    tag = "ge" if ge else "gt"
    workdir.mkdir(parents=True, exist_ok=True)
    mp = workdir / f"toy_refresh_param_{tag}.imi"
    pp = workdir / "bad.imiprop"
    mp.write_text(model, encoding="utf-8", newline="\n")
    pp.write_text("property := #synth EF(loc[ctrl] = bad);\n", encoding="utf-8", newline="\n")

    def wsl_path(p: Path) -> str:
        s = str(p.resolve()).replace("\\", "/")
        return "/mnt/" + s[0].lower() + s[2:]
    cmd = ["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "--", "bash", "-lc",
           f"cd '{wsl_path(workdir)}' && /usr/bin/time -f 'WALL %e' /usr/local/bin/imitator "
           f"'{mp.name}' bad.imiprop -output-prefix out_{tag}"]
    try:
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        out = r.stdout + r.stderr
        res_files = sorted(workdir.glob(f"out_{tag}*.res"))
        res = res_files[-1].read_text(encoding="utf-8", errors="replace") if res_files else ""
        return {"ok": r.returncode == 0 and bool(res), "wall_s": round(time.time() - t0, 2),
                "stdout_tail": out[-1500:], "res": res}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": repr(e)}


def eval_imitator_constraint(res: str, T: int, W: int, R: int) -> bool | None:
    """Evaluate the synthesized EF(bad) constraint at an integer point (True = bad)."""
    m = re.search(r"BEGIN CONSTRAINT(.*?)END CONSTRAINT", res, re.S)
    if not m:
        return None
    txt = m.group(1).strip()
    if txt in ("False", "false"):
        return False
    disj = [d.strip() for d in re.split(r"\bOR\b", txt)]
    env = {"T": T, "W": W, "R": R}
    for d in disj:
        ok = True
        for atom in [a.strip() for a in d.split("&")]:
            if atom in ("True", "true", ""):
                continue
            expr = atom.replace("=", "==").replace(">==", ">=").replace("<==", "<=")
            expr = re.sub(r"(\d)\s*([TWR])", r"\1*\2", expr)
            try:
                if not eval(expr, {"__builtins__": {}}, env):  # noqa: S307 (own output)
                    ok = False
                    break
            except Exception:  # noqa: BLE001
                return None
        if ok:
            return True
    return False


# ---------------------------------------------------------------- (b) monitor
def monitor_IR(o: np.ndarray, R: int) -> np.ndarray:
    """Counter of consecutive CLOSED frames saturating at R-1 (ceil(log2 R) bits);
    alarm at frame t iff frames t-R+1..t are all CLOSED."""
    alarm = np.zeros(len(o), dtype=bool)
    c = 0
    for t, ot in enumerate(o):
        if ot:
            c = 0
        else:
            if c >= R - 1:
                alarm[t] = True        # c stays saturated at R-1
            else:
                c += 1
    return alarm


def brute_violations(o: np.ndarray, R: int) -> np.ndarray:
    """v[t] = all CLOSED in window [t-R+1, t]."""
    closed = (~o).astype(int)
    cs = np.concatenate([[0], np.cumsum(closed)])
    v = np.zeros(len(o), dtype=bool)
    for t in range(R - 1, len(o)):
        v[t] = cs[t + 1] - cs[t + 1 - R] == R
    return v


def shield(o_policy: np.ndarray, R: int) -> np.ndarray:
    """Force OPEN when R-1 consecutive CLOSED frames precede t (enforces I_R)."""
    out = o_policy.copy()
    c = 0
    for t in range(len(out)):
        if not out[t] and c >= R - 1:
            out[t] = True
        c = 0 if out[t] else c + 1
    return out


# ---------------------------------------------------------------- (c) mutants
MUTANT_CLASSES = {
    "M1_boundary": [
        ("M1a", "reuse cap fsl > W (as deployed)", dict(cap_op="gt")),
        ("M1b", "reuse cap fsl >= W+1", dict(cap_op="ge", cap_off=1)),
        ("M1c", "reuse cap fsl > W-1 (over-strict)", dict(cap_op="gt", cap_off=-1)),
        ("M1d", "refresh pos % T == 1", dict(rf_phase=1)),
        ("M1e", "gate opens on score > tau (strict)", dict(tau_strict=True)),
        ("M1f", "hysteresis close on score < tau_close (strict)", dict(close_strict=True)),
    ],
    "M2_counter_reset": [
        ("M2a", "fsl not reset by refresh OPENs", dict(reset_on_refresh=False)),
        ("M2b", "fsl not reset by gate OPENs", dict(reset_on_gate=False)),
        ("M2c", "fsl reset to 1 instead of 0", dict(reset_val=1)),
        ("M2d", "fsl reset on gate REQUEST even when frame invalid", dict(reset_on_request=True)),
        ("M2e", "watchdog counter not reset when gate opens", dict(wd_no_reset=True)),
        ("M2f", "fsl reset only by cap OPENs", dict(reset_on_refresh=False, reset_on_gate=False)),
    ],
    "M3_init": [
        ("M3a", "fsl_0 = 0 (first frame not forced)", dict(fsl0=0)),
        ("M3b", "fsl_0 = W-1", dict(fsl0="W-1")),
        ("M3c", "refresh fires at t = 0 too", dict(rf_at0=True)),
        ("M3d", "hysteresis initial state open", dict(h0=True)),
        ("M3e", "watchdog counter starts at 11", dict(cc0=11)),
        ("M3f", "refresh phase offset T/2", dict(rf_phase="T//2")),
    ],
    "M4_refresh_phase": [
        ("M4a", "refresh relative to last OPEN (not absolute)", dict(rf_relative=True)),
        ("M4b", "refresh period T+1", dict(T_off=1)),
        ("M4c", "refresh period 2T", dict(T_mul=2)),
        ("M4d", "refresh suppressed if gate open in last frame", dict(rf_suppress_gate=True)),
        ("M4e", "refresh disabled", dict(rf_off=True)),
        ("M4f", "refresh only on valid even frames", dict(rf_even=True)),
    ],
    "M5_clock_unit": [
        ("M5a", "fsl counts in 2-frame ticks (frame_step 2)", dict(fsl_step=0.5)),
        ("M5b", "W given at 15 fps applied at 30 fps (W doubled)", dict(W_mul=2)),
        ("M5c", "T given at 15 fps applied at 30 fps (T doubled)", dict(T_mul=2)),
        ("M5d", "fsl counts only frames with motion score > 0.3", dict(fsl_motion_only=True)),
        ("M5e", "fsl saturates at W (never exceeds)", dict(fsl_sat="W")),
        ("M5f", "fsl incremented twice per CLOSED frame", dict(fsl_step=2)),
    ],
    "M6_hysteresis_watchdog": [
        ("M6a", "tau_close = tau_open (no hysteresis)", dict(tau_close=0.55)),
        ("M6b", "watchdog interval 7", dict(wd_int=7)),
        ("M6c", "watchdog trigger 13", dict(wd_trig=13)),
        ("M6d", "watchdog disabled", dict(wd_off=True)),
        ("M6e", "tau_open 0.60", dict(tau_open=0.60)),
        ("M6f", "watchdog fires only once per closed run", dict(wd_once=True)),
    ],
}
QUICK10 = ["M1a", "M1b", "M2a", "M2c", "M3a", "M4a", "M5a", "M5e", "M6b", "M6d"]


def run_mutant(score: np.ndarray, valid: np.ndarray, T: int, W: int, m: dict) -> np.ndarray:
    """Configurable controller (fixed '>=' baseline when m == {})."""
    tau_open = m.get("tau_open", 0.55)
    tau_close = m.get("tau_close", 0.30)
    n = len(score)
    # hysteresis (with optional strictness / init)
    h = np.zeros(n, dtype=bool)
    st = m.get("h0", False)
    for t, sc in enumerate(score.astype(np.float64)):
        op = sc > tau_open if m.get("tau_strict") else sc >= tau_open
        cl = sc < tau_close if m.get("close_strict") else sc <= tau_close
        if op:
            st = True
        elif cl:
            st = False
        h[t] = st
    Teff = T * m.get("T_mul", 1) + m.get("T_off", 0)
    Weff = W * m.get("W_mul", 1)
    fsl0 = m.get("fsl0", Weff + 1)
    if fsl0 == "W-1":
        fsl0 = Weff - 1
    ph_off = m.get("rf_phase", 0)
    if ph_off == "T//2":
        ph_off = Teff // 2
    wd_trig, wd_int = m.get("wd_trig", 12), m.get("wd_int", 6)
    o = np.zeros(n, dtype=bool)
    cc, fsl, since_rf = m.get("cc0", 0), float(fsl0), 0
    wd_fired = False
    prev_gate = False
    for t in range(n):
        if not valid[t]:
            fsl += 1
            continue
        if h[t]:
            if not m.get("wd_no_reset"):
                cc = 0
            wd_fired = False
        else:
            cc += 1
        wd = (not h[t]) and cc >= wd_trig and cc % wd_int == 0 and not m.get("wd_off")
        if m.get("wd_once"):
            wd = wd and not wd_fired
            wd_fired = wd_fired or wd
        g = bool(h[t]) or wd
        # refresh
        if m.get("rf_off"):
            rf = False
        elif m.get("rf_relative"):
            rf = fsl + 1 >= Teff
        else:
            rf = ((t > 0) or m.get("rf_at0", False)) and (t - ph_off) % Teff == 0
            if m.get("rf_even"):
                rf = rf and t % 2 == 0
        if m.get("rf_suppress_gate") and prev_gate:
            rf = False
        thr = Weff + m.get("cap_off", 0)
        fsl_cmp = math.floor(fsl)
        cap = (fsl_cmp > thr) if m.get("cap_op", "ge") == "gt" else (fsl_cmp >= thr)
        o[t] = g or rf or cap
        prev_gate = g
        reset = o[t]
        if o[t] and rf and not g and not cap and m.get("reset_on_refresh", True) is False:
            reset = False
        if o[t] and g and not cap and not rf and m.get("reset_on_gate", True) is False:
            reset = False
        if m.get("reset_on_request") and g:
            reset = True
        if reset:
            fsl = float(m.get("reset_val", 0))
        else:
            if m.get("fsl_motion_only") and score[t] <= 0.3:
                pass
            else:
                fsl += m.get("fsl_step", 1)
            if m.get("fsl_sat") == "W":
                fsl = min(fsl, float(Weff))
    return o


def metrics(o: np.ndarray, events: list[tuple[int, int]], f0: int, R: int) -> dict:
    """mAP-proxy = % of in-event frames at or after the event's first OPEN;
    energy = Deep3 linear model; I_R violations = windows of R CLOSED frames."""
    covered = total = 0
    for on, D in events:
        p = on - f0
        w = o[p:p + D]
        idx = np.flatnonzero(w)
        total += D
        covered += (D - idx[0]) if len(idx) else 0
    a = float(o.mean())
    return {"cov": covered, "tot": total, "a": a,
            "E": E_IDLE + a * (E_ACTIVE - E_IDLE),
            "viol": int(brute_violations(o, R).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=15)
    ap.add_argument("--W", type=int, default=4)
    ap.add_argument("--grid-T", type=int, default=30)
    ap.add_argument("--grid-W", type=int, default=20)
    ap.add_argument("--n-videos", type=int, default=20)
    ap.add_argument("--no-imitator", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    R = min(a.T, a.W + 1)
    out = {"seed": a.seed, "T": a.T, "W": a.W, "R": R}

    # (a) ------------------------------------------------------------------
    t0 = time.time()
    mism = {"ge": [], "gt": []}
    n_pts = 0
    for T in range(1, a.grid_T + 1):
        for W in range(0, a.grid_W + 1):
            n_pts += 1
            for ge in (True, False):
                g, _ = max_gap_explicit(T, W, ge)
                expect = min(T, W + 1) if ge else min(T, W + 2)
                if g != expect:
                    mism["ge" if ge else "gt"].append((T, W, g, expect))
    g_gt, wit_gt = max_gap_explicit(a.T, a.W, False)
    g_ge, wit_ge = max_gap_explicit(a.T, a.W, True)
    out["a_region_bruteforce"] = {
        "grid": f"T in [1,{a.grid_T}], W in [0,{a.grid_W}]", "n_points": n_pts,
        "mismatch_ge_vs_min(T,W+1)": mism["ge"], "mismatch_gt_vs_min(T,W+2)": mism["gt"],
        "max_gap_deployed_T15_W4": g_gt, "max_gap_fixed_T15_W4": g_ge,
        "witness_deployed_OPEN_CLOSED": "".join("O" if x else "." for x in wit_gt),
        "wall_s": round(time.time() - t0, 2)}
    print("(a) brute force:", {k: (len(v) if isinstance(v, list) else v)
                               for k, v in out["a_region_bruteforce"].items()})
    if not a.no_imitator:
        imi = {}
        wd = TOPIC_DIR / "results" / "imitator_p0"
        for ge in (True, False):
            r = run_imitator(15, 10, 20, ge, wd)
            tag = "ge" if ge else "gt"
            if r.get("ok"):
                bad_mismatch, checked = [], 0
                for T in range(1, 16):
                    for W in range(0, 11):
                        for Rc in range(1, 21):
                            bad = eval_imitator_constraint(r["res"], T, W, Rc)
                            if bad is None:
                                continue
                            checked += 1
                            true_bad = Rc < (min(T, W + 1) if ge else min(T, W + 2))
                            if bad != true_bad:
                                bad_mismatch.append((T, W, Rc, bad))
                m = re.search(r"BEGIN CONSTRAINT(.*?)END CONSTRAINT", r["res"], re.S)
                imi[tag] = {"ok": True, "wall_s": r["wall_s"], "n_points_checked": checked,
                            "n_mismatch": len(bad_mismatch), "mismatch_head": bad_mismatch[:10],
                            "constraint": m.group(1).strip() if m else None}
            else:
                imi[tag] = {"ok": False, "detail": r.get("error") or r.get("stdout_tail")}
            print(f"(a) IMITATOR {tag}:", {k: v for k, v in imi[tag].items() if k != "constraint"})
        out["a_region_imitator"] = imi

    # (b) ------------------------------------------------------------------
    res_b = {}
    for var in ("gt", "ge"):
        fr = pd.read_parquet(OUT_ROOT / f"traces_{var}.parquet")
        n_win = n_det = n_false = 0
        delays, n_closed_runs_geR = [], 0
        extra, frames = 0, 0
        shield_viol = 0
        for (_, _), x in fr.groupby(["dataset", "video"], sort=False):
            o = x.o_t.to_numpy()
            al = monitor_IR(o, R)
            bv = brute_violations(o, R)
            n_win += int(bv.sum())
            n_det += int((al & bv).sum())
            n_false += int((al & ~bv).sum())
            # delay from first frame of each violating closed run to first alarm
            closed = ~o
            t = 0
            while t < len(o):
                if closed[t]:
                    s = t
                    while t < len(o) and closed[t]:
                        t += 1
                    if t - s >= R:
                        n_closed_runs_geR += 1
                        first_alarm = s + int(np.argmax(al[s:t]))
                        delays.append(first_alarm - s + 1)
                else:
                    t += 1
            so = shield(o, R)
            extra += int(so.sum() - o.sum())
            frames += len(o)
            shield_viol += int(brute_violations(so, R).sum())
        res_b[var] = {"n_frames": frames, "n_violating_windows": n_win,
                      "n_detected_windows": n_det, "n_false_alarms": n_false,
                      "n_violating_closed_runs": n_closed_runs_geR,
                      "detect_delay_frames_max": int(max(delays)) if delays else None,
                      "detect_delay_frames_min": int(min(delays)) if delays else None,
                      "shield_extra_activation": round(extra / frames, 6),
                      "shield_extra_bound_1_over_R": round(1 / R, 6),
                      "shield_residual_violations": shield_viol}
        print(f"(b) {var}:", res_b[var])
    # randomized differential test of the monitor (synthetic traces)
    n_traces, n_disagree, max_delay = 2000, 0, 0
    for _ in range(n_traces):
        p_open = rng.uniform(0.02, 0.6)
        o = rng.random(int(rng.integers(20, 400))) < p_open
        al, bv = monitor_IR(o, R), brute_violations(o, R)
        n_disagree += int((al != bv).sum())
    res_b["random_diff_test"] = {"n_traces": n_traces, "seed": a.seed,
                                 "n_frame_disagreements": n_disagree,
                                 "monitor_state_bits": math.ceil(math.log2(R))}
    print("(b) random:", res_b["random_diff_test"])
    out["b_monitor"] = res_b

    # (c) ------------------------------------------------------------------
    all_m = {mid: (cls, desc, kw) for cls, lst in MUTANT_CLASSES.items() for mid, desc, kw in lst}
    gates = pd.read_parquet(GATE_TRACE)
    keys = [(ds, jp.stem) for ds in CORPUS for jp in sorted((DET_ROOT / ds).glob("*.json"))]
    pick = sorted(rng.choice(len(keys), size=a.n_videos, replace=False).tolist())
    vids = [keys[i] for i in pick]
    data = []
    for ds, key in vids:
        video = key.split("__", 1)[1]
        fa, has = load_detections(ds, key)
        gv = gates[(gates.dataset == ds) & (gates.video == video)].sort_values("frame")
        data.append((ds, video, fa[0], gv.score.to_numpy(), gv.valid.to_numpy().astype(bool),
                     extract_events(fa, has)))
    # sanity: baseline mutant engine == policy_ref fixed stream
    ref_ge = pd.read_parquet(OUT_ROOT / "traces_ge.parquet")
    base_mismatch = 0
    for ds, video, f0, sc, va, ev in data:
        ob = run_mutant(sc, va, a.T, a.W, {})
        rr = ref_ge[(ref_ge.dataset == ds) & (ref_ge.video == video)].o_t.to_numpy()
        base_mismatch += int((ob != rr).sum())

    def agg(m: dict) -> dict:
        cov = tot = viol = nfr = 0
        asum = 0.0
        vids_viol = 0
        for ds, video, f0, sc, va, ev in data:
            o = run_mutant(sc, va, a.T, a.W, m)
            r = metrics(o, ev, f0, R)
            cov += r["cov"]
            tot += r["tot"]
            viol += r["viol"]
            vids_viol += int(r["viol"] > 0)
            asum += r["a"] * len(o)
            nfr += len(o)
        aa = asum / nfr
        return {"mAP_proxy_pct": 100 * cov / tot, "E_mJ": E_IDLE + aa * (E_ACTIVE - E_IDLE),
                "a": aa, "IR_violating_windows": viol, "videos_with_violation": vids_viol}
    base = agg({})
    rows = []
    for mid in QUICK10:
        cls, desc, kw = all_m[mid]
        r = agg(kw)
        d_map = r["mAP_proxy_pct"] - base["mAP_proxy_pct"]
        d_e = 100 * (r["E_mJ"] - base["E_mJ"]) / base["E_mJ"]
        inv = abs(d_map) < 0.5 and abs(d_e) < 1.0 and r["IR_violating_windows"] > 0
        rows.append({"mutant": mid, "class": cls, "description": desc,
                     "dmAP_proxy_pts": round(d_map, 3), "dE_pct": round(d_e, 3),
                     "IR_violating_windows": r["IR_violating_windows"],
                     "videos_with_violation": r["videos_with_violation"],
                     "violates_IR": r["IR_violating_windows"] > 0,
                     "benchmark_invisible": inv})
    df = pd.DataFrame(rows)
    df.to_csv(TOPIC_DIR / "results" / "p0_mutants_quick.csv", index=False)
    print(df.to_string(index=False))
    out["c_mutants"] = {
        "n_classes": len(MUTANT_CLASSES), "n_listed": len(all_m),
        "listed": {cls: [f"{mid}: {d}" for mid, d, _ in lst] for cls, lst in MUTANT_CLASSES.items()},
        "quick_run": {"n_mutants": len(QUICK10), "n_videos": a.n_videos,
                      "videos": [f"{d}/{v}" for d, v, *_ in data],
                      "baseline_engine_vs_policy_ref_frame_mismatch": base_mismatch,
                      "baseline": {k: round(v, 4) if isinstance(v, float) else v
                                   for k, v in base.items()},
                      "thresholds": {"dmAP_proxy_pts": 0.5, "dE_pct": 1.0},
                      "n_violate_IR": int(df.violates_IR.sum()),
                      "n_benchmark_invisible": int(df.benchmark_invisible.sum()),
                      "rows": rows}}
    (TOPIC_DIR / "results" / "p0_sanity.json").write_text(json.dumps(out, indent=1, default=str),
                                                         encoding="utf-8")


if __name__ == "__main__":
    main()
