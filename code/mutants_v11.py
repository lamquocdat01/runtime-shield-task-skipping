"""mutants_v11.py -- P2 Viec 2 (item 5): the PREREG v1.1 mutant set (>= 40 NEW mutants) and
their step semantics, written from spec/FSM_06.md (no Deep3/ASMAG code).

Two families, both first-order w.r.t. their original unless marked 2nd-order:
  G1 family (N01..N41): mutants of the fixed '>=' gate controller G1 (original = ORIG, the
    same original as v1.0). Targets (CP1 #4): reuse cap (comparison, reset on the wrong event,
    counter overflow / saturation, cap in time instead of frames), the fsl counter (step k,
    only some frames counted), clock unit / kappa (wall-clock refresh with kappa != 1, stale
    register / 1-tick delay), missing frames (valid = 0: reset / no reset / double count,
    phantom OPEN, deferred refresh, catch-up), refresh phase (+/- offsets, period, locked to
    the last gate OPEN).
  CB family (C01..C10): mutants of the composition G1 + ASMAG circuit breaker (original =
    ORIG_CB: G1 '>=' with the correct breaker, spec/FSM_06.md sec. 2: W_cb = 300,
    theta_high 0.95, theta_low 0.60, K = 300, P = 30; activation = gate request of the last
    G1-stepped frame, i.e. the previous frame's activation as in ASMAG; d = 0, so every pure
    BYPASS frame is dispatched). Targets: probe schedule (slip, length, period), exit/entry
    handling of the frozen fsl, probes without detection, half-rate bypass (the d = 2 analogue
    of Proposition 2.7), probe on a missing frame.
Every mutant is a dict of knobs; the empty dict is the original. `Ctl11` extends
mutants.Ctl (v1.0 semantics are reproduced exactly for v1.0 knobs: `selftest`).
Model checking (I_R, adversarial score from the 7 representative values of mutants.SCORES,
all frames valid, b = 0, d = 0):
  G1 family: explicit-state BFS over (canonical controller state, closed run) -- mc_g1.
  CB family: BFS over ACTIVE frames, PROBE frames and pure-BYPASS stretches executed as one
    macro step (G1 frozen, refresh clock advanced); ACTIVE -> BYPASS entry and the exit
    decision at the end of a probe are adversarial (over-approximation of the 300-frame
    window and of the probe rate) -- mc_cb.
Exploratory (not a hypothesis): mc_g1 with a frame-drop budget b = 1 per non-dispatch run at
R = 6 = R*(ORIG, b = 1) (Theorem 2'), so the missing-frame mutants get a verdict.
Usage: python code/mutants_v11.py --seed 42   (prints the list + runs the self-tests)
"""
from __future__ import annotations

import argparse
import math
import time
from collections import deque

import numpy as np
from numba import njit

from mutants import ALL as ALL_V10, SCORES, Ctl

K_CB, P_CB, WIN_CB, TH_HI, TH_LO = 300, 30, 300, 0.95, 0.60

# ------------------------------------------------------------------------- the mutant list
# (id, family, class, description, knobs)
MUTANTS_V11 = [
    # --- reuse cap: comparison / counter width / saturation / time units
    ("N01", "G1", "cap", "cap fsl == W (equality instead of >=)", dict(cap_eq=True)),
    ("N02", "G1", "cap", "fsl stored in a 3-bit counter (mod 8)", dict(fsl_mod=8)),
    ("N03", "G1", "cap", "fsl stored in a 2-bit counter (mod 4)", dict(fsl_mod=4)),
    ("N04", "G1", "cap", "fsl saturates at W - 1", dict(fsl_sat="W-1")),
    ("N05", "G1", "cap", "cap in ms: fsl*33.3 >= W*41.7 (frame period mis-set, ratio 0.8)",
     dict(cap_time=0.8)),
    ("N06", "G1", "cap", "cap in ms with ratio 0.6 (W given at 25 fps, run at 15 fps)",
     dict(cap_time=0.6)),
    ("N07", "G1", "cap", "cap in ms with ratio 1.25 (over-strict)", dict(cap_time=1.25)),
    ("N08", "G1", "cap", "fsl also reset when the hysteresis gate closes", dict(reset_on_close=True)),
    ("N09", "G1", "cap", "fsl reset to -1 after an OPEN", dict(reset_val=-1)),
    ("N10", "G1", "cap", "fsl reset to 2 after an OPEN", dict(reset_val=2)),
    ("N11", "G1", "cap", "cap OPENs do not reset fsl", dict(reset_on_cap=False)),
    ("N12", "G1", "cap", "gate OPENs reset fsl only on the rising edge of the gate",
     dict(reset_gate_rise_only=True)),
    ("N19", "G1", "cap", "alternative fix: fsl + 1 > W", dict(pre_inc="gt")),
    ("N20", "G1", "cap", "pre-increment: fsl + 1 >= W (over-strict)", dict(pre_inc="ge")),
    ("N39", "G1", "cap", "OPENs held by hysteresis (score below tau_open) do not reset fsl",
     dict(hold_no_reset=True)),
    ("N40", "G1", "cap", "watchdog OPENs do not reset fsl", dict(wd_no_reset_fsl=True)),
    ("N41", "G1", "cap", "cap compares fsl > W - 1 after a refresh only, else fsl > W",
     dict(cap_after_rf_only=True)),
    # --- fsl counter
    ("N13", "G1", "counter", "fsl += 0.75 per CLOSED frame (floor)", dict(fsl_step=0.75)),
    ("N14", "G1", "counter", "fsl += 0.9 per CLOSED frame (floor)", dict(fsl_step=0.9)),
    ("N15", "G1", "counter", "fsl += 1.5 per CLOSED frame (floor)", dict(fsl_step=1.5)),
    ("N16", "G1", "counter", "fsl counts only frames with score > 0.1", dict(fsl_count_thr=0.1)),
    ("N17", "G1", "counter", "fsl counts only frames with score > 0.55", dict(fsl_count_thr=0.55)),
    ("N18", "G1", "counter", "fsl counts only static frames (score <= 0.30)",
     dict(fsl_count_static=True)),
    ("N27", "G1", "counter", "fsl counts wall-clock ticks at kappa = 0.8", dict(fsl_step=0.8)),
    # --- clock unit / kappa / delay
    ("N21", "G1", "clock", "cap reads the previous frame's fsl (stale register, 1 tick)",
     dict(cap_stale=True)),
    ("N22", "G1", "clock", "OPEN decision applied one frame late (pipeline register)",
     dict(decision_delay1=True)),
    ("N23", "G1", "clock", "refresh on a wall clock at kappa = 1/2", dict(rf_kappa=(1, 2))),
    ("N24", "G1", "clock", "refresh on a wall clock at kappa = 4/5", dict(rf_kappa=(4, 5))),
    ("N25", "G1", "clock", "refresh on a wall clock at kappa = 5/4", dict(rf_kappa=(5, 4))),
    ("N26", "G1", "clock", "refresh on a wall clock at kappa = 2", dict(rf_kappa=(2, 1))),
    # --- refresh phase
    ("N28", "G1", "phase", "refresh one tick early (pos % T == T - 1)", dict(rf_phase=-1)),
    ("N29", "G1", "phase", "refresh phase offset 7", dict(rf_phase=7)),
    ("N30", "G1", "phase", "refresh locked to the last gate OPEN (every T frames after it)",
     dict(rf_rel_gate=True)),
    ("N31", "G1", "phase", "refresh period T - 1 (modulo off-by-one)", dict(T_off=-1)),
    ("N32", "G1", "phase", "refresh OPENs reset fsl to -1", dict(rf_reset_val=-1)),
    # --- missing frames (latent on the corpus: valid = 1 everywhere)
    ("N33", "G1", "missing", "missing frame resets fsl to 0", dict(inv_mode="reset")),
    ("N34", "G1", "missing", "missing frame does not increment fsl", dict(inv_mode="noinc")),
    ("N35", "G1", "missing", "missing frame increments fsl twice", dict(inv_mode="double")),
    ("N36", "G1", "missing", "after a missing frame the next valid frame is forced OPEN",
     dict(inv_mode="catchup")),
    ("N37", "G1", "missing", "phantom OPEN: a refresh/cap due on a missing frame resets fsl",
     dict(inv_mode="phantom")),
    ("N38", "G1", "missing", "refresh due on a missing frame deferred to the next valid frame",
     dict(inv_mode="defer")),
    # --- CB family (original ORIG_CB)
    ("C01", "CB", "cb", "fsl reset to 0 when BYPASS is left", dict(cb_exit_reset=True)),
    ("C02", "CB", "cb", "probe frames step G1 but the detector is not run",
     dict(cb_probe_nodetect=True)),
    ("C03", "CB", "cb", "the BYPASS entry frame is not dispatched", dict(cb_entry_drop=True)),
    ("C04", "CB", "cb", "half-rate bypass (detector every 2nd pure-BYPASS frame)",
     dict(cb_half=True)),
    ("C05", "CB", "cb", "2nd-order: half-rate bypass + probe length 31 (`< 0` off-by-one)",
     dict(cb_half=True, cb_P=31)),
    ("C06", "CB", "cb", "2nd-order: half-rate bypass + probe every 301 frames",
     dict(cb_half=True, cb_K=301)),
    ("C07", "CB", "cb", "exit possible after every probe frame (running probe rate)",
     dict(cb_exit_any=True)),
    ("C08", "CB", "cb", "probe slip: probe starts at bypass age % K == 1", dict(cb_slip=1)),
    ("C09", "CB", "cb", "probe skipped when its first frame is missing",
     dict(cb_probe_inv_skip=True)),
    ("C10", "CB", "cb", "G1 refresh phase counts only G1-stepped frames (frozen in BYPASS)",
     dict(cb_rf_frozen=True)),
]
ALL_V11 = {mid: (fam, cls, desc, kw) for mid, fam, cls, desc, kw in MUTANTS_V11}
assert len(ALL_V11) == len(MUTANTS_V11) >= 40


# --------------------------------------------------------------------------- G1 controller
class Ctl11(Ctl):
    """mutants.Ctl + the v1.1 knobs. State tuple:
    (tm, first, st, cc, wd_fired, prev_gate, fsl, prev_fsl, pend, cg, prev_h, flag, rf_last)"""

    def __init__(self, T: int, W: int, m: dict):
        super().__init__(T, W, m)
        q = m.get("rf_kappa", (1, 1))[1]
        self.period = math.lcm(self.Teff * q, 2)

    def init(self):
        tm, first, st, cc, wd, pg, fsl = super().init()
        return (tm, first, st, cc, wd, pg, fsl, fsl, False, 0, st, False, False)

    def gate_of(self, s) -> bool:
        return bool(s[5])

    def advance(self, s, L: int):
        """L frames in which G1 is not stepped (pure BYPASS): only the clock moves."""
        return ((s[0] + L) % self.period, False) + s[2:]

    def step(self, s, score: float, valid: bool = True):
        m = self.m
        tm, first, st, cc, wd_fired, prev_gate, fsl, prev_fsl, pend, cg, prev_h, flag, rf_last = s
        t_pos = 0 if first else tm
        op_ = score > self.tau_open if m.get("tau_strict") else score >= self.tau_open
        cl_ = score < self.tau_close if m.get("close_strict") else score <= self.tau_close
        if op_:
            st = True
        elif cl_:
            st = False
        h = st
        ntm = (tm + 1) % self.period
        if not valid:
            inv = m.get("inv_mode")
            nfsl = fsl + 1
            nflag = flag
            if inv == "reset":
                nfsl = 0.0
            elif inv == "noinc":
                nfsl = fsl
            elif inv == "double":
                nfsl = fsl + 2
            elif inv == "catchup":
                nflag = True
            elif inv in ("phantom", "defer"):
                rf = ((not first) or m.get("rf_at0", False)) and (t_pos - self.ph) % self.Teff == 0
                cap = math.floor(fsl) >= self.Weff + m.get("cap_off", 0)
                if inv == "phantom" and (rf or cap):
                    nfsl = 0.0
                if inv == "defer" and rf:
                    nflag = True
            ncg = cg + 1
            out = pend if m.get("decision_delay1") else False
            npend = False if m.get("decision_delay1") else pend
            return (bool(out), (ntm, False, st, cc, wd_fired, prev_gate, float(nfsl), fsl, npend,
                                ncg, h, nflag, False))
        if h:
            if not m.get("wd_no_reset"):
                cc = 0
            wd_fired = False
        else:
            cc += 1
        wd = (not h) and cc >= self.wd_trig and cc % self.wd_int == 0 and not m.get("wd_off")
        if m.get("wd_once"):
            wd = wd and not wd_fired
            wd_fired = wd_fired or wd
        g = bool(h) or wd
        # refresh
        if m.get("rf_off"):
            rf = False
        elif m.get("rf_relative"):
            rf = fsl + 1 >= self.Teff
        elif "rf_kappa" in m:
            p, q = m["rf_kappa"]
            rf = (not first) and (p * t_pos) % (q * self.Teff) < p
        elif m.get("rf_rel_gate"):
            rf = cg > 0 and cg % self.Teff == 0
        else:
            rf = ((not first) or m.get("rf_at0", False)) and (t_pos - self.ph) % self.Teff == 0
            if m.get("rf_even"):
                rf = rf and t_pos % 2 == 0
        if m.get("rf_suppress_gate") and prev_gate:
            rf = False
        forced = False
        if m.get("inv_mode") in ("catchup", "defer") and flag:
            if m["inv_mode"] == "defer":
                rf = True
            else:
                forced = True
        # reuse cap
        thr = self.Weff + m.get("cap_off", 0)
        fv = prev_fsl if m.get("cap_stale") else fsl
        fc = math.floor(fv)
        if m.get("cap_eq"):
            cap = fc == thr
        elif "cap_time" in m:
            cap = fc * m["cap_time"] >= thr - 1e-9
        elif m.get("pre_inc") == "gt":
            cap = fc + 1 > thr
        elif m.get("pre_inc") == "ge":
            cap = fc + 1 >= thr
        elif m.get("cap_after_rf_only"):
            cap = (fc > thr - 1) if rf_last else (fc > thr)
        else:
            cap = (fc > thr) if m.get("cap_op", "ge") == "gt" else (fc >= thr)
        o = g or rf or cap or forced
        prev_gate = g
        reset = o
        only_rf = o and rf and not g and not cap
        only_g = o and g and not cap and not rf
        if only_rf and m.get("reset_on_refresh", True) is False:
            reset = False
        if only_g and m.get("reset_on_gate", True) is False:
            reset = False
        if m.get("reset_on_request") and g:
            reset = True
        if o and cap and not g and not rf and m.get("reset_on_cap", True) is False:
            reset = False
        if only_g and m.get("reset_gate_rise_only") and not (h and not prev_h):
            reset = False
        if only_g and m.get("hold_no_reset") and h and not op_:
            reset = False
        if o and wd and not h and not cap and not rf and m.get("wd_no_reset_fsl"):
            reset = False
        old_fsl = fsl
        if reset:
            fsl = float(m["rf_reset_val"]) if (only_rf and "rf_reset_val" in m) \
                else float(m.get("reset_val", 0))
        else:
            inc = True
            if m.get("fsl_motion_only") and score <= 0.3:
                inc = False
            if "fsl_count_thr" in m and not score > m["fsl_count_thr"]:
                inc = False
            if m.get("fsl_count_static") and not score <= 0.30:
                inc = False
            if inc:
                fsl += m.get("fsl_step", 1)
            if m.get("fsl_sat") == "W":
                fsl = min(fsl, float(self.Weff))
            elif m.get("fsl_sat") == "W-1":
                fsl = min(fsl, float(self.Weff - 1))
            if "fsl_mod" in m:
                fsl = float(fsl % m["fsl_mod"])
        if m.get("reset_on_close") and prev_h and not h and not o:
            fsl = 0.0
        ncg = 0 if g and o else cg + 1
        if m.get("decision_delay1"):
            out, pend = pend, o
        else:
            out = o
        return (bool(out), (ntm, False, st, cc, wd_fired, prev_gate, fsl, old_fsl, pend, ncg, h,
                            False, bool(o and rf)))

    def _canon(self, s):
        tm, first, st, cc, wd_fired, prev_gate, fsl, prev_fsl, pend, cg, prev_h, flag, rfl = s
        if cc >= self.wd_trig:
            cc = self.wd_trig + (cc - self.wd_trig) % self.wd_int
        B = float(3 * (self.Weff + abs(self.m.get("cap_off", 0)) + self.Teff) + 5)
        if "fsl_mod" not in self.m:
            fsl, prev_fsl = min(fsl, B), min(prev_fsl, B)
        if cg >= self.Teff:
            cg = self.Teff + (cg - self.Teff) % self.Teff
        return (tm, first, st, cc, wd_fired, prev_gate, fsl, prev_fsl, pend, cg, prev_h, flag, rfl)


def run_trace11(score: np.ndarray, valid: np.ndarray, T: int, W: int, m: dict) -> np.ndarray:
    c = Ctl11(T, W, m)
    s = c.init()
    o = np.zeros(len(score), dtype=bool)
    for t in range(len(score)):
        o[t], s = c.step(s, float(score[t]), bool(valid[t]))
    return o


# --------------------------------------------------------------- G1 + circuit breaker (CB)
def cb_params(m: dict) -> tuple[int, int]:
    return m.get("cb_K", K_CB), m.get("cb_P", P_CB)


def run_trace_cb(score: np.ndarray, valid: np.ndarray, T: int, W: int, m: dict) -> np.ndarray:
    """Corpus replay of G1 ('>=', knobs of m that concern G1 are none for the CB family) in
    series with the ASMAG breaker (spec/FSM_06.md sec. 2), with the CB mutation knobs."""
    K, P = cb_params(m)
    slip = m.get("cb_slip", 0)
    c = Ctl11(T, W, {})
    s = c.init()
    n = len(score)
    o = np.zeros(n, dtype=bool)
    acts = deque(maxlen=WIN_CB)
    state, age, in_probe, left, pacts = "A", 0, False, 0, []
    a_prev = 0.0
    prev_disp = True
    for t in range(n):
        sc, va = float(score[t]), bool(valid[t])
        phase = "active"
        if state == "A":
            acts.append(a_prev)
            if len(acts) >= WIN_CB and sum(acts) / len(acts) > TH_HI:
                state, age, in_probe, pacts = "B", 0, False, []
                phase = "entry"
        else:
            age += 1
            if not in_probe and age % K == slip % K:
                if not (m.get("cb_probe_inv_skip") and not va):
                    in_probe, left, pacts = True, P, []
            phase = "probe" if in_probe else "bypass"
        if phase in ("active", "probe"):
            ot, s = c.step(s, sc, va)
            if phase == "probe" and m.get("cb_probe_nodetect"):
                ot = False
            a_prev = 1.0 if c.gate_of(s) else 0.0
            if phase == "probe":
                pacts.append(a_prev)
                left -= 1
                rate = sum(pacts) / len(pacts)
                if left <= 0 or m.get("cb_exit_any"):
                    if left <= 0:
                        in_probe = False
                    if rate < TH_LO:
                        state, in_probe = "A", False
                        acts = deque(pacts[-WIN_CB:], maxlen=WIN_CB)
                        age = 0
                        if m.get("cb_exit_reset"):
                            s = s[:6] + (0.0,) + s[7:]
        else:
            if not m.get("cb_rf_frozen"):
                s = c.advance(s, 1)
            if phase == "entry" and m.get("cb_entry_drop"):
                ot = False
            elif m.get("cb_half"):
                ot = va and not prev_disp
            else:
                ot = va
        o[t] = ot
        prev_disp = bool(ot)
    return o


@njit(cache=True)
def controller_cb(score, valid, vstart, T, W, tau_open, tau_close, wd_trig, wd_int, K, P,
                  win, th_hi, th_lo):
    """numba: correct G1 ('>=', hysteresis + watchdog inline, frozen in pure BYPASS) + correct
    breaker; same semantics as run_trace_cb with m = {} (checked in selftest/p2_t4)."""
    n = len(score)
    o = np.zeros(n, dtype=np.bool_)
    ring = np.zeros(win, dtype=np.float64)
    pbuf = np.zeros(P + 1, dtype=np.float64)
    for v in range(len(vstart) - 1):
        st = False
        cc = 0
        fsl = W + 1
        tm = 0
        first = True
        state = 0
        age = 0
        in_probe = False
        left = 0
        psum = 0.0
        pcnt = 0
        a_prev = 0.0
        rsum = 0.0
        rlen = 0
        rpos = 0
        for t in range(vstart[v], vstart[v + 1]):
            phase = 0                      # 0 active, 1 probe, 2 bypass/entry
            if state == 0:
                if rlen == win:
                    rsum -= ring[rpos]
                else:
                    rlen += 1
                ring[rpos] = a_prev
                rsum += a_prev
                rpos = (rpos + 1) % win
                if rlen >= win and rsum / rlen > th_hi:
                    state = 1
                    age = 0
                    in_probe = False
                    psum = 0.0
                    pcnt = 0
                    phase = 2
            else:
                age += 1
                if (not in_probe) and age % K == 0:
                    in_probe = True
                    left = P
                    psum = 0.0
                    pcnt = 0
                phase = 1 if in_probe else 2
            if phase == 2:
                tm += 1
                first = False
                o[t] = valid[t]
                continue
            # G1 step (fixed >=)
            sc = score[t]
            if sc >= tau_open:
                st = True
            elif sc <= tau_close:
                st = False
            if not valid[t]:
                fsl += 1
                tm += 1
                first = False
                ot = False
                g = a_prev > 0.5
            else:
                if st:
                    cc = 0
                else:
                    cc += 1
                wd = (not st) and cc >= wd_trig and cc % wd_int == 0
                g = st or wd
                rf = (not first) and tm % T == 0
                cap = fsl >= W
                ot = g or rf or cap
                fsl = 0 if ot else fsl + 1
                tm += 1
                first = False
            o[t] = ot
            a_prev = 1.0 if g else 0.0
            if phase == 1:
                pbuf[pcnt] = a_prev
                psum += a_prev
                pcnt += 1
                left -= 1
                if left <= 0:
                    in_probe = False
                    if psum / pcnt < th_lo:
                        state = 0
                        age = 0
                        # reseed the rolling window with the probe activations (P < win)
                        rsum = 0.0
                        rlen = 0
                        rpos = 0
                        for k in range(pcnt):
                            ring[rpos] = pbuf[k]
                            rsum += pbuf[k]
                            rpos = (rpos + 1) % win
                            rlen += 1
    return o


# --------------------------------------------------------------------------- model checking
def mc_g1(ctl, R: int, b: int = 0, max_states: int = 3_000_000) -> dict:
    """BFS over (canonical controller state, closed run, drops in the run). b > 0 lets the
    adversary drop up to b frames per non-dispatch run (run-budget semantics, exploratory)."""
    t0 = time.perf_counter()
    s0 = (ctl._canon(ctl.init()), 0, 0)
    parent = {s0: None}
    dq = deque([s0])
    while dq:
        key = dq.popleft()
        s, run, dr = key
        for valid in ((True, False) if (b > 0 and dr < b) else (True,)):
            for sc in (SCORES if valid else SCORES[:1]):
                o, ns = ctl.step(s, sc, valid)
                nrun, ndr = (0, 0) if o else (run + 1, dr + (0 if valid else 1))
                nk = (ctl._canon(ns), nrun, ndr)
                if nk in parent:
                    continue
                parent[nk] = (key, o)
                if nrun >= R:
                    wit, k = [], nk
                    while parent[k] is not None:
                        k, ok = parent[k]
                        wit.append("O" if ok else ".")
                    return {"violates": True, "witness": "".join(reversed(wit)),
                            "n_states": len(parent), "seconds": time.perf_counter() - t0}
                if len(parent) > max_states:
                    return {"violates": None, "n_states": len(parent),
                            "seconds": time.perf_counter() - t0, "note": "state cap"}
                dq.append(nk)
    return {"violates": False, "witness": None, "n_states": len(parent),
            "seconds": time.perf_counter() - t0}


def mc_cb(T: int, W: int, m: dict, R: int, max_states: int = 3_000_000) -> dict:
    """I_R for G1 + breaker. Nodes: ('A', s, run) ACTIVE frame next; ('P', s, left, run, rate)
    PROBE frame next (rate = probe activations so far, only tracked for C07 where an early
    exit needs it -- otherwise the exit decision is free). Pure-BYPASS stretches are macro
    steps (G1 frozen; refresh clock advanced unless C10)."""
    t0 = time.perf_counter()
    K, P = cb_params(m)
    slip = m.get("cb_slip", 0)
    first_len = (slip % K) if slip % K else K     # pure frames from entry to the first probe
    c = Ctl11(T, W, {})

    def stretch(s, run, L, entry):
        """L pure-BYPASS frames; returns (s', run', witness, violated)."""
        w = []
        prev_disp = run == 0
        for i in range(L):
            if entry and i == 0 and m.get("cb_entry_drop"):
                d = False
            elif m.get("cb_half"):
                d = not prev_disp
            else:
                d = True
            run = 0 if d else run + 1
            prev_disp = d
            w.append("B" if d else "b")
            if run >= R:
                return s, run, "".join(w), True
        s2 = s if m.get("cb_rf_frozen") else c.advance(s, L)
        return s2, run, "".join(w), False

    root = ("A", c._canon(c.init()), 0)
    parent = {root: None}
    dq = deque([root])

    def push(nk, par, lab):
        if nk in parent:
            return False
        parent[nk] = (par, lab)
        dq.append(nk)
        return True

    def witness(k, tail=""):
        out = [tail]
        while parent[k] is not None:
            k, lab = parent[k]
            out.append(lab)
        return "".join(reversed(out))

    while dq:
        node = dq.popleft()
        if node[0] == "A":
            _, s, run = node
            for sc in SCORES:
                o, ns = c.step(s, sc)
                nrun = 0 if o else run + 1
                if nrun >= R:
                    return {"violates": True, "witness": witness(node, "O" if o else "."),
                            "n_states": len(parent), "seconds": time.perf_counter() - t0}
                push(("A", c._canon(ns), nrun), node, "O" if o else ".")
            # enter BYPASS at this frame: `first_len` pure frames, then a probe
            s2, r2, w, bad = stretch(s, run, first_len, True)
            if bad:
                return {"violates": True, "witness": witness(node, w), "n_states": len(parent),
                        "seconds": time.perf_counter() - t0}
            push(("P", c._canon(s2), P, r2, 0), node, w)
        else:
            _, s, left, run, acc = node
            for sc in SCORES:
                o, ns = c.step(s, sc)
                if m.get("cb_probe_nodetect"):
                    o = False
                nrun = 0 if o else run + 1
                lab = "P" if o else "p"
                if nrun >= R:
                    return {"violates": True, "witness": witness(node, lab),
                            "n_states": len(parent), "seconds": time.perf_counter() - t0}
                nacc = acc + (1 if c.gate_of(ns) else 0) if m.get("cb_exit_any") else 0
                nleft = left - 1
                cs = c._canon(ns)
                can_exit = nleft == 0 or (m.get("cb_exit_any") and nacc < TH_LO * (P - nleft))
                if can_exit:
                    es = cs[:6] + (0.0,) + cs[7:] if m.get("cb_exit_reset") else cs
                    push(("A", es, nrun), node, lab + "|")
                if nleft > 0:
                    push(("P", cs, nleft, nrun, nacc), node, lab)
                else:
                    s2, r2, w, bad = stretch(cs, nrun, K - P, False)
                    if bad:
                        return {"violates": True, "witness": witness(node, lab + w),
                                "n_states": len(parent), "seconds": time.perf_counter() - t0}
                    push(("P", c._canon(s2), P, r2, 0), node, lab + w)
            if len(parent) > max_states:
                return {"violates": None, "n_states": len(parent),
                        "seconds": time.perf_counter() - t0, "note": "state cap"}
    return {"violates": False, "witness": None, "n_states": len(parent),
            "seconds": time.perf_counter() - t0}


# -------------------------------------------------------------------------------- selftest
def selftest(seed: int = 42) -> dict:
    """Implementation checks on SYNTHETIC score sequences only (no corpus, no MC verdicts of
    mutants): (1) Ctl11 == mutants.Ctl for every v1.0 mutant; (2) each v1.1 knob changes the
    trace of some synthetic sequence or is documented as latent; (3) run_trace_cb(m={}) ==
    controller_cb on synthetic sequences with long active stretches."""
    from mutants import run_trace
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(6):
        n = 2000
        base = rng.choice(np.array(SCORES), n)
        burst = rng.random(n) < 0.02
        sc = np.where(np.cumsum(burst) % 3 == 0, base, rng.choice(np.array([0.6, 1.0]), n))
        seqs.append(sc)
    va = np.ones(2000, dtype=bool)
    out = {"v10_equal": 0, "v10_total": 0}
    for mid, (_, _, kw) in ALL_V10.items():
        for sc in seqs:
            out["v10_total"] += 1
            out["v10_equal"] += bool(np.array_equal(run_trace(sc, va, 15, 4, kw),
                                                    run_trace11(sc, va, 15, 4, kw)))
    base_tr = [run_trace11(sc, va, 15, 4, {}) for sc in seqs]
    changed = {}
    inv = va.copy()
    inv[rng.choice(2000, 40, replace=False)] = False
    for mid, (fam, cls, desc, kw) in ALL_V11.items():
        if fam != "G1":
            continue
        diff = any(not np.array_equal(run_trace11(sc, va, 15, 4, kw), b)
                   for sc, b in zip(seqs, base_tr))
        if cls == "missing":
            diff = any(not np.array_equal(run_trace11(sc, inv, 15, 4, kw),
                                          run_trace11(sc, inv, 15, 4, {})) for sc in seqs)
        changed[mid] = bool(diff)
    out["g1_knob_changes_synthetic_trace"] = changed
    # CB: python replay vs numba on synthetic sequences with long active stretches
    eq = 0
    for k in range(4):
        n = 3000
        sc = np.where((np.arange(n) // (400 + 100 * k)) % 2 == 1,
                      rng.choice(np.array([0.6, 1.0, 0.57]), n),
                      rng.choice(np.array(SCORES), n))
        vs = np.array([0, n], dtype=np.int64)
        a = run_trace_cb(sc, np.ones(n, dtype=bool), 15, 4, {})
        b = controller_cb(sc, np.ones(n, dtype=np.bool_), vs, 15, 4, 0.55, 0.30, 12, 6,
                          K_CB, P_CB, WIN_CB, TH_HI, TH_LO)
        eq += bool(np.array_equal(a, b))
    out["cb_python_eq_numba_synthetic"] = f"{eq}/4"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    for mid, fam, cls, desc, kw in MUTANTS_V11:
        print(f"{mid}  {fam:3s} {cls:8s} {desc}")
    print(len(MUTANTS_V11), "mutants")
    print(selftest(a.seed))


if __name__ == "__main__":
    main()
