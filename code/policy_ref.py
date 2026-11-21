"""policy_ref.py -- independent reference re-implementation of the refresh-bounded
frame-skipping gate controller of Deep3, written from spec/FSM_06.md (no code imported
or copied from the Deep3 / ASMAG repositories).

Subject system (cited, not copied):
  Deep3 = "certifiable frame skipping", IEEE Internet of Things Journal 2026,
  DOI 10.1109/JIOT.2026.3740931, Sec. II ("R-refresh property", R = min(T_refresh,
  W_maxskip + 1), T_refresh = 15, W_maxskip = 4 -> R = 5) and Sec. VI-E ("A deployed
  system leaks the certificate": the deployed reuse cap is
  `frames_since_last_open > W_maxskip`, period 6 instead of R = 5; `>=` restores it).

Per frame t of a video (0-based position), with S_t the logged motion-gate score:
  h_t       = 1 if S_t >= TAU_OPEN, 0 if S_t <= TAU_CLOSE, else h_{t-1}   (h_{-1} = 0)
  wd_t      = (not h_t) and cc_t >= WD_TRIGGER and cc_t % WD_INTERVAL == 0
              (cc_t = consecutive frames with h = 0, counted including t)
  g_t       = h_t or wd_t                                 (gate request)
  refresh_t = t > 0 and t % T_refresh == 0
  cap_t     = fsl_t >  W_maxskip   (variant "gt", as deployed)
            | fsl_t >= W_maxskip   (variant "ge", fixed)
  o_t       = valid_t and (g_t or refresh_t or cap_t)  (OPEN = detector runs;
              a missing frame, valid_t = 0, is never OPEN)
  fsl_{t+1} = 0 if o_t else fsl_t + 1,    fsl_0 = W_maxskip + 1
Events: maximal runs of consecutive frames (in TEMPORAL order) with >= 1 detection.
An event (onset s, duration D) is missed iff o_t = 0 for all t in [s, s + D).

Inputs (read-only, under THS_DATASETS, default ./datasets):
  _derived/detections_cpu_yolo26s/<DS>/<key>.json  -- always-on detections (event GT proxy)
  _derived/06/inputs/gate_scores.parquet            -- per-frame gate score S_t, valid_t,
                                                       in_event (cross-check only)
     (copied from the Deep3 corrected re-run, LASIESTA already in temporal order; see
      _derived/06/inputs/PROVENANCE.json)
LASIESTA detection JSONs index frames by position in the LEXICOGRAPHIC file list
(`I_BS_01-1, -10, -100, ...`); they are re-sorted here by the numeric frame number
in the file name (0-based = number - 1) before events are built.

Usage:
  python code/policy_ref.py --seed 42
Outputs:
  _derived/06/traces_gt.parquet, _derived/06/traces_ge.parquet  (per-frame streams)
  _derived/06/events_gt.parquet, _derived/06/events_ge.parquet  (per-event outcomes)
  results/p0_trace_stats.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

TOPIC_DIR = Path(__file__).resolve().parent.parent
DATASETS = Path(os.environ.get("THS_DATASETS", "datasets"))
DET_ROOT = DATASETS / "_derived" / "detections_cpu_yolo26s"
OUT_ROOT = DATASETS / "_derived" / "06"
GATE_TRACE = OUT_ROOT / "inputs" / "gate_scores.parquet"
CORPUS = ("CDnet2014", "BMC", "LASIESTA")  # 53 + 20 + 48 = 121 videos (VIRAT has no events)

# ASMAG gate watchdog constants (asmag-trc configs/..._cdnet_full.yaml, p4_asmag_plus)
TAU_OPEN = 0.55
TAU_CLOSE = 0.30
WD_TRIGGER = 12
WD_INTERVAL = 6


def lasiesta_order(video: str) -> np.ndarray:
    """Map lexicographic position -> 0-based temporal frame for a LASIESTA video."""
    files = sorted(p.name for p in (DATASETS / "LASIESTA" / video).glob("*.bmp"))
    nums = np.array([int(re.search(r"-(\d+)\.bmp$", f).group(1)) for f in files])
    return nums - 1


def load_detections(ds: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (frame_abs, has_det) in temporal order."""
    with open(DET_ROOT / ds / f"{key}.json", encoding="utf-8") as f:
        frames = json.load(f)["frames"]
    idx = np.array([fr["frame"] for fr in frames])
    has = np.array([len(fr.get("boxes", [])) > 0 for fr in frames])
    if ds == "LASIESTA":
        video = key.split("__", 1)[1]
        order = lasiesta_order(video)
        if len(order) != len(idx):
            raise RuntimeError(f"{key}: {len(order)} bmp files vs {len(idx)} JSON frames")
        idx = order[idx]
        srt = np.argsort(idx, kind="stable")
        idx, has = idx[srt], has[srt]
    if not np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))):
        raise RuntimeError(f"{ds}/{key}: frame indices not contiguous after ordering")
    return idx, has


def extract_events(frame_abs: np.ndarray, has: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of consecutive detection frames -> [(onset_abs, duration)]."""
    ev, start = [], None
    for i, h in enumerate(has):
        if h and start is None:
            start = i
        elif not h and start is not None:
            ev.append((int(frame_abs[start]), i - start))
            start = None
    if start is not None:
        ev.append((int(frame_abs[start]), len(has) - start))
    return ev


def hysteresis(score: np.ndarray, tau_open: float = TAU_OPEN,
               tau_close: float = TAU_CLOSE) -> np.ndarray:
    h = np.zeros(len(score), dtype=bool)
    state = False
    for t, sc in enumerate(score.astype(np.float64)):
        if sc >= tau_open:
            state = True
        elif sc <= tau_close:
            state = False
        h[t] = state
    return h


def run_policy(h: np.ndarray, valid: np.ndarray, T_refresh: int, W_maxskip: int,
               variant: str) -> dict:
    """Simulate the controller over one video. Returns per-frame arrays."""
    n = len(h)
    o = np.zeros(n, dtype=bool)
    g = np.zeros(n, dtype=bool)
    wd = np.zeros(n, dtype=bool)
    rf = np.zeros(n, dtype=bool)
    cap = np.zeros(n, dtype=bool)
    fsl_arr = np.zeros(n, dtype=np.int32)
    cc, fsl = 0, W_maxskip + 1
    for t in range(n):
        if not valid[t]:          # missing frame: no gate pass, never OPEN
            fsl_arr[t] = fsl
            fsl += 1
            continue
        if h[t]:
            cc = 0
        else:
            cc += 1
            wd[t] = cc >= WD_TRIGGER and cc % WD_INTERVAL == 0
        g[t] = bool(h[t]) or wd[t]
        rf[t] = t > 0 and t % T_refresh == 0
        cap[t] = (fsl > W_maxskip) if variant == "gt" else (fsl >= W_maxskip)
        fsl_arr[t] = fsl
        o[t] = g[t] or rf[t] or cap[t]
        fsl = 0 if o[t] else fsl + 1
    return {"o": o, "g": g, "wd": wd, "refresh": rf, "cap": cap, "fsl": fsl_arr}


def max_open_gap(o: np.ndarray) -> int:
    idx = np.flatnonzero(o)
    return int(np.max(np.diff(idx))) if len(idx) > 1 else len(o)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="unused (deterministic); recorded")
    ap.add_argument("--T", type=int, default=15, help="T_refresh")
    ap.add_argument("--W", type=int, default=4, help="W_maxskip")
    ap.add_argument("--out-json", default=str(TOPIC_DIR / "results" / "p0_trace_stats.json"))
    a = ap.parse_args()
    np.random.seed(a.seed)
    R = min(a.T, a.W + 1)

    gates = pd.read_parquet(GATE_TRACE)
    frames_out = {"gt": [], "ge": []}
    events_out = {"gt": [], "ge": []}
    n_videos = 0
    n_in_event_mismatch = 0
    for ds in CORPUS:
        for jp in sorted((DET_ROOT / ds).glob("*.json")):
            key = jp.stem
            video = key.split("__", 1)[1]
            fa, has = load_detections(ds, key)
            gv = gates[(gates.dataset == ds) & (gates.video == video)].sort_values("frame")
            if len(gv) != len(fa) or gv.frame.iloc[0] != fa[0]:
                raise RuntimeError(f"{ds}/{video}: gate trace {len(gv)} frames vs det {len(fa)}")
            h = hysteresis(gv.score.to_numpy())
            valid = gv.valid.to_numpy().astype(bool)
            n_in_event_mismatch += int((gv.in_event.to_numpy().astype(bool) != has).sum())
            n_videos += 1
            evs = extract_events(fa, has)
            for var in ("gt", "ge"):
                s = run_policy(h, valid, a.T, a.W, var)
                frames_out[var].append(pd.DataFrame({
                    "dataset": ds, "video": video, "frame": fa.astype(np.int32),
                    "in_event": has, "score": gv.score.to_numpy(), "h_t": h,
                    "wd": s["wd"], "g_t": s["g"], "refresh": s["refresh"], "cap": s["cap"],
                    "fsl": s["fsl"], "o_t": s["o"]}))
                o = s["o"]
                for k, (on, D) in enumerate(evs):
                    p = on - fa[0]
                    win = o[p:p + D]
                    opens = np.flatnonzero(win)
                    events_out[var].append({
                        "dataset": ds, "video": video, "event_id": k, "onset": on,
                        "duration": D, "missed": int(len(opens) == 0),
                        "latency": int(opens[0]) if len(opens) else -1,
                        "caught_by_gate": int(bool(h[p:p + D].any()))})

    stats = {"seed": a.seed, "T_refresh": a.T, "W_maxskip": a.W, "R": R,
             "tau_open": TAU_OPEN, "tau_close": TAU_CLOSE,
             "watchdog": {"trigger": WD_TRIGGER, "interval": WD_INTERVAL},
             "n_videos": n_videos,
             "n_frames_in_event_mismatch_vs_deep3_dump": n_in_event_mismatch,
             "variants": {}}
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for var in ("gt", "ge"):
        fr = pd.concat(frames_out[var], ignore_index=True)
        ev = pd.DataFrame(events_out[var])
        fp, ep = OUT_ROOT / f"traces_{var}.parquet", OUT_ROOT / f"events_{var}.parquet"
        fr.to_parquet(fp, index=False)
        ev.to_parquet(ep, index=False)
        gaps = fr.groupby(["dataset", "video"]).o_t.apply(lambda s: max_open_gap(s.to_numpy()))
        # gap distribution over all consecutive OPEN pairs
        all_gaps = np.concatenate([np.diff(np.flatnonzero(x.to_numpy()))
                                   for _, x in fr.groupby(["dataset", "video"]).o_t])
        dge = ev[ev.duration >= R]
        leaks = dge[dge.missed == 1]
        stats["variants"][var] = {
            "comparison": "fsl > W_maxskip" if var == "gt" else "fsl >= W_maxskip",
            "n_frames": int(len(fr)), "n_events": int(len(ev)),
            "n_events_by_dataset": ev.groupby("dataset").size().astype(int).to_dict(),
            "activation_a": round(float(fr.o_t.mean()), 6),
            "max_open_gap": int(gaps.max()),
            "n_videos_with_gap_gt_R": int((gaps > R).sum()),
            "n_open_gaps_total": int(len(all_gaps)),
            "n_open_gaps_eq_R_plus_1": int((all_gaps == R + 1).sum()),
            "miss_all": int(ev.missed.sum()),
            "n_events_D_ge_R": int(len(dge)),
            "miss_D_ge_R": int(len(leaks)),
            "miss_D_ge_R_plus_1": int(dge[(dge.duration >= R + 1)].missed.sum()),
            "leaked_events": leaks[["dataset", "video", "onset", "duration"]].assign(
                last=lambda d: d.onset + d.duration - 1).to_dict("records"),
            "traces_parquet": str(fp), "traces_sha256": sha256(fp),
            "events_parquet": str(ep), "events_sha256": sha256(ep)}
    stats["gate_trace_input"] = {"path": str(GATE_TRACE), "sha256": sha256(GATE_TRACE)}
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1, default=int)
    for var, v in stats["variants"].items():
        print(f"[{var}] n_videos={n_videos} events={v['n_events']} max_gap={v['max_open_gap']} "
              f"miss D>=R: {v['miss_D_ge_R']}/{v['n_events_D_ge_R']}  leaks={v['leaked_events']}")


if __name__ == "__main__":
    main()
