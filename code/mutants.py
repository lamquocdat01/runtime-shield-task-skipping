"""mutants.py -- P1 Viec 3: the 36 controller mutants (6 classes x 6, listed in P0,
code/p0_sanity.py MUTANT_CLASSES) as ONE step function shared by
  (1) corpus replay  (run_trace: score sequence -> OPEN/CLOSED trace), and
  (2) explicit-state model checking of I_R (model_check: adversarial score sequence).
The step semantics are those of p0_sanity.run_mutant (baseline = fixed '>=' controller);
`check_equal_to_p0` verifies frame-by-frame equality on the corpus.

Model checking abstraction: the score S_t is adversarial. Every predicate of the controller
compares S_t with one of 0.30, 0.55, 0.60 (strict or not), so the 7 representative values
SCORES below realise every combination of predicate outcomes: the explicit-state search over
these inputs is exact for the controller (all frames valid, b = 0, d = 0: the deployed
setting). State is finite after saturating fsl and the watchdog counter (see _canon).
"""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from p0_sanity import MUTANT_CLASSES  # noqa: F401  (the 36 mutants, frozen at P0)

SCORES = (0.0, 0.30, 0.42, 0.55, 0.57, 0.60, 1.0)
ALL = {mid: (cls, desc, kw) for cls, lst in MUTANT_CLASSES.items() for mid, desc, kw in lst}


class Ctl:
    def __init__(self, T: int, W: int, m: dict):
        self.m = m
        self.tau_open = m.get("tau_open", 0.55)
        self.tau_close = m.get("tau_close", 0.30)
        self.Teff = T * m.get("T_mul", 1) + m.get("T_off", 0)
        self.Weff = W * m.get("W_mul", 1)
        fsl0 = m.get("fsl0", self.Weff + 1)
        self.fsl0 = float(self.Weff - 1 if fsl0 == "W-1" else fsl0)
        ph = m.get("rf_phase", 0)
        self.ph = self.Teff // 2 if ph == "T//2" else ph
        self.wd_trig, self.wd_int = m.get("wd_trig", 12), m.get("wd_int", 6)
        self.period = self.Teff * 2 // math.gcd(self.Teff, 2)

    def init(self):
        # (t_mod, first, h, cc, wd_fired, prev_gate, fsl)
        return (0, True, bool(self.m.get("h0", False)), self.m.get("cc0", 0), False, False,
                self.fsl0)

    def step(self, s, score: float, valid: bool = True):
        m = self.m
        tm, first, st, cc, wd_fired, prev_gate, fsl = s
        t_pos = 0 if first else tm          # only t mod period matters for t > 0
        op_ = score > self.tau_open if m.get("tau_strict") else score >= self.tau_open
        cl_ = score < self.tau_close if m.get("close_strict") else score <= self.tau_close
        if op_:
            st = True
        elif cl_:
            st = False
        h = st
        if not valid:
            nfsl = fsl + 1
            return (False, ((tm + 1) % self.period, False, st, cc, wd_fired, prev_gate, nfsl))
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
        if m.get("rf_off"):
            rf = False
        elif m.get("rf_relative"):
            rf = fsl + 1 >= self.Teff
        else:
            rf = ((not first) or m.get("rf_at0", False)) and (t_pos - self.ph) % self.Teff == 0
            if m.get("rf_even"):
                rf = rf and t_pos % 2 == 0
        if m.get("rf_suppress_gate") and prev_gate:
            rf = False
        thr = self.Weff + m.get("cap_off", 0)
        fc = math.floor(fsl)
        cap = (fc > thr) if m.get("cap_op", "ge") == "gt" else (fc >= thr)
        o = g or rf or cap
        prev_gate = g
        reset = o
        if o and rf and not g and not cap and m.get("reset_on_refresh", True) is False:
            reset = False
        if o and g and not cap and not rf and m.get("reset_on_gate", True) is False:
            reset = False
        if m.get("reset_on_request") and g:
            reset = True
        if reset:
            fsl = float(m.get("reset_val", 0))
        else:
            if not (m.get("fsl_motion_only") and score <= 0.3):
                fsl += m.get("fsl_step", 1)
            if m.get("fsl_sat") == "W":
                fsl = min(fsl, float(self.Weff))
        return (bool(o), ((tm + 1) % self.period, False, st, cc, wd_fired, prev_gate, fsl))

    def _canon(self, s):
        """Finite abstraction preserving all future behaviour."""
        tm, first, st, cc, wd_fired, prev_gate, fsl = s
        if cc >= self.wd_trig:
            cc = self.wd_trig + (cc - self.wd_trig) % self.wd_int
        fsl = min(fsl, float(max(self.Weff + abs(self.m.get("cap_off", 0)), self.Teff) + 3))
        return (tm, first, st, cc, wd_fired, prev_gate, fsl)


def run_trace(score: np.ndarray, valid: np.ndarray, T: int, W: int, m: dict) -> np.ndarray:
    c = Ctl(T, W, m)
    s = c.init()
    o = np.zeros(len(score), dtype=bool)
    for t in range(len(score)):
        o[t], s = c.step(s, float(score[t]), bool(valid[t]))
    return o


def model_check(T: int, W: int, m: dict, R: int, max_states: int = 2_000_000) -> dict:
    """Exhaustive BFS over (controller state, closed-run counter) with adversarial scores.
    Returns verdict, witness (OPEN/CLOSED string) if violated, states, seconds."""
    t0 = time.perf_counter()
    c = Ctl(T, W, m)
    s0 = (c._canon(c.init()), 0)
    parent = {s0: None}
    dq = deque([s0])
    while dq:
        s, run = dq.popleft()
        for sc in SCORES:
            o, ns = c.step(s, sc)
            nrun = 0 if o else run + 1
            key = (c._canon(ns), nrun)
            if key in parent:
                continue
            parent[key] = ((s, run), o, sc)
            if nrun >= R:
                wit = []
                k = key
                while parent[k] is not None:
                    pk, ok, _ = parent[k]
                    wit.append("O" if ok else ".")
                    k = pk
                return {"violates": True, "witness": "".join(reversed(wit)),
                        "n_states": len(parent), "seconds": time.perf_counter() - t0}
            if len(parent) > max_states:
                return {"violates": None, "n_states": len(parent),
                        "seconds": time.perf_counter() - t0, "note": "state cap"}
            dq.append(key)
    return {"violates": False, "witness": None, "n_states": len(parent),
            "seconds": time.perf_counter() - t0}
