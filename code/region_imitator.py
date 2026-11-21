"""region_imitator.py -- P1 Viec 1 (item 5a): parametric synthesis of the extended certificate
region with IMITATOR 3.4 (WSL Ubuntu-24.04), and check of every synthesised constraint against
the exhaustive explicit-state engine (code/region_bruteforce.py) on the integer points of the
parameter box.

For each combination and comparison (>= fixed, > deployed) the template model/gate_ext.imi is
instantiated and `#synth EF(loc[ctrl] = bad)` is run. The result is the set of parameter
valuations where the certificate FAILS; the region K is its complement in the box.
Outputs:
  results/region_ext/imi/<combo>_<cmp>.imi, .res       (model instance and IMITATOR output)
  results/region_ext/<combo>.res                       (concatenated .res of the combo)
  results/region_ext/imitator_summary.json             (soundness, time, integer-point check)
Usage:
  python code/region_imitator.py --seed 42 [--only base,d_block]
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import pandas as pd

TOPIC_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = TOPIC_DIR / "results" / "region_ext"
IMI_DIR = OUT_DIR / "imi"
TEMPLATE = TOPIC_DIR / "model" / "gate_ext.imi"

# box of the parameters (IMITATOR synthesises over rationals; checked on integers)
BOX = {"TMAX": 12, "WMAX": 8, "RMAX": 24, "DMAX": 4, "BMAX": 3}
# smaller box for the combinations whose synthesis does not finish in 20 min on BOX
SMALL = {"TMAX": 8, "WMAX": 6, "RMAX": 16, "DMAX": 3, "BMAX": 2}

# name -> (tags, kappa list)
K1 = [Fraction(1)]
KALL = [Fraction(1, 2), Fraction(2, 3), Fraction(3, 4), Fraction(4, 3), Fraction(3, 2),
        Fraction(2)]
COMBOS = {
    "base": ({"NOBLOCK"}, K1),
    "d_pipe": ({"D", "NOBLOCK"}, K1),
    "d_block": ({"D", "BLOCK"}, K1),
    "b": ({"B", "NOBLOCK"}, K1),
    "d_b_pipe": ({"D", "B", "NOBLOCK"}, K1),
    "d_b_block": ({"D", "B", "BLOCK"}, K1, SMALL),
    "kappa": ({"NOBLOCK"}, KALL),
    "d_kappa_b_block": ({"D", "B", "BLOCK"}, KALL, SMALL),
    "CB": ({"CB", "NOBLOCK"}, K1, SMALL),
    "CB_d_pipe": ({"CB", "D", "NOBLOCK"}, K1, SMALL),
    "CB_d_block": ({"CB", "D", "BLOCK"}, K1, SMALL),
}


def instantiate(tags: set[str], ge: bool, kappa: Fraction, box: dict) -> str:
    tags = set(tags) | ({"NOCB"} if "CB" not in tags else set())
    src = TEMPLATE.read_text(encoding="utf-8")
    out = []
    for line in src.splitlines():
        m = re.search(r"\(\*@([A-Z@*]+)\*\)\s*$", line)
        if m:
            need = [t for t in m.group(1).split("*@") if t]
            if not all(t in tags for t in need):
                continue
            line = line[:m.start()].rstrip()
        # inline optional fragments [[TAG: text]]
        line = re.sub(r"\[\[([A-Z]+): ([^\]]*)\]\]",
                      lambda mm: mm.group(2) if mm.group(1) in tags else "", line)
        out.append(line)
    txt = "\n".join(out) + "\n"
    P, Q, c = kappa.numerator, kappa.denominator, 0 if ge else 1
    hasd = "D" in tags

    def lin(coefs: dict, const: int) -> str:
        """IMITATOR linear term sum(k * v) + const."""
        parts = [(f"{k} * {v}" if k != 1 else v) for v, k in coefs.items() if k]
        if const or not parts:
            parts.append(str(const))
        return " + ".join(parts).replace("+ -", "- ")
    rep = {"@P@": str(P), "@Q@": str(Q), "@CAPOFF@": str(c),
           "@TQ@": lin({"T": Q}, 0),
           "@BAD@": lin({"R": P, "d": -P} if hasd else {"R": P}, P),
           "@CAPLIM@": lin({"W": P}, P * c),
           "@CAPSAT@": lin({"W": P}, P * (c + 1)),
           "@XCMAX@": lin({"W": P}, P * (c + 2)),
           "@XSMAX@": lin({"R": P}, P),
           "@DP@": lin({"d": P}, 0)}
    rep.update({f"@{k}@": str(v) for k, v in box.items()})
    for k, v in rep.items():
        txt = txt.replace(k, v)
    return txt


def wsl_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:]


def run_imitator(model_txt: str, stem: str, timeout: int) -> dict:
    IMI_DIR.mkdir(parents=True, exist_ok=True)
    mp = IMI_DIR / f"{stem}.imi"
    pp = IMI_DIR / "bad.imiprop"
    mp.write_text(model_txt, encoding="utf-8", newline="\n")
    pp.write_text("property := #synth EF(loc[ctrl] = bad);\n", encoding="utf-8", newline="\n")
    for old in IMI_DIR.glob(f"{stem}*.res"):
        old.unlink()
    cmd = ["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "--", "bash", "-lc",
           f"cd '{wsl_path(IMI_DIR)}' && timeout {timeout} /usr/local/bin/imitator "
           f"'{mp.name}' bad.imiprop -output-prefix {stem}"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.time() - t0
    res_p = IMI_DIR / f"{stem}.res"
    res = res_p.read_text(encoding="utf-8", errors="replace") if res_p.exists() else ""
    info = {"stem": stem, "returncode": r.returncode, "wall_s": round(wall, 2),
            "res_file": str(res_p.relative_to(TOPIC_DIR)) if res else None}
    for key, pat in [("soundness", r"Constraint soundness\s*:\s*(.+)"),
                     ("termination", r"Termination\s*:\s*(.+)"),
                     ("n_states", r"Number of states\s*:\s*(\d+)"),
                     ("time_s", r"Total computation time\s*:\s*([\d.]+)")]:
        m = re.search(pat, res)
        info[key] = m.group(1).strip() if m else None
    m = re.search(r"BEGIN CONSTRAINT(.*?)END CONSTRAINT", res, re.S)
    info["constraint"] = m.group(1).strip() if m else None
    if not res:
        info["stderr_tail"] = (r.stdout + r.stderr)[-800:]
    return info


_num = re.compile(r"(?<![A-Za-z_])(\d+(?:/\d+)?)")


def compile_constraint(txt: str):
    """Constraint text -> list of disjuncts, each a list of python expressions."""
    if txt is None:
        return None
    if txt.strip() in ("False", "false"):
        return []
    disj = []
    for d in re.split(r"\bOR\b", txt):
        atoms = []
        for a in d.split("&"):
            a = a.strip()
            if a in ("", "True", "true"):
                continue
            a = re.sub(r"(?<![<>=!])=(?!=)", "==", a)
            a = _num.sub(lambda m: f"F('{m.group(1)}')", a)
            a = re.sub(r"\)\s*([A-Za-z_])", r")*\1", a)       # 2 T -> 2*T
            atoms.append(a)
        disj.append(atoms)
    return disj


def eval_constraint(disj, env: dict) -> bool:
    g = {"F": Fraction, "__builtins__": {}}
    for atoms in disj:
        if all(eval(a, g, env) for a in atoms):  # noqa: S307 (own tool output)
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="unused (exhaustive); recorded")
    ap.add_argument("--only", default="")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    grid = pd.read_parquet(OUT_DIR / "bruteforce_grid.parquet")
    summ_path = OUT_DIR / "imitator_summary.json"
    summary = json.loads(summ_path.read_text(encoding="utf-8")) if summ_path.exists() else {}
    summary.update({"seed": a.seed, "box": BOX, "property": "#synth EF(loc[ctrl] = bad)",
                    "imitator": "3.4 Cheese Durian (WSL Ubuntu-24.04)"})
    summary.setdefault("runs", {})
    names = [n for n in COMBOS if not a.only or n in a.only.split(",")]
    for name in names:
        tags, kaps = COMBOS[name][:2]
        box = COMBOS[name][2] if len(COMBOS[name]) > 2 else BOX
        res_txt = []
        for kap, ge in itertools.product(kaps, (True, False)):
            stem = f"{name}_k{kap.numerator}-{kap.denominator}_{'ge' if ge else 'gt'}"
            info = run_imitator(instantiate(tags, ge, kap, box), stem, a.timeout)
            if info["res_file"]:
                res_txt.append((IMI_DIR / f"{stem}.res").read_text(encoding="utf-8",
                                                                   errors="replace"))
            # integer-point check against the explicit-state engine
            disj = compile_constraint(info["constraint"])
            chk = {"n_points": 0, "n_mismatch": 0, "mismatch_head": []}
            if disj is not None:
                block = 1 if "BLOCK" in tags else 0
                sub = grid[(grid["ge"] == int(ge)) & (grid.kappa_str == str(kap))
                           & (grid.cb == int("CB" in tags))
                           & (grid["T"] <= box["TMAX"]) & (grid.W <= box["WMAX"])]
                sub = sub[(sub.d <= (box["DMAX"] if "D" in tags else 0))
                          & (sub.b <= (box["BMAX"] if "B" in tags else 0))]
                sub = sub[(sub.block == block) | (sub.d <= 1)]
                for row in sub.itertuples(index=False):
                    for R in range(1, box["RMAX"] + 1):
                        env = {"T": row.T, "W": row.W, "R": R, "d": row.d, "b": row.b}
                        bad_imi = eval_constraint(disj, env)
                        bad_true = R < row.R_star
                        chk["n_points"] += 1
                        if bad_imi != bad_true:
                            chk["n_mismatch"] += 1
                            if len(chk["mismatch_head"]) < 10:
                                chk["mismatch_head"].append({**env, "imi_bad": bad_imi,
                                                             "R_star": int(row.R_star)})
            info["integer_check"] = chk
            info["box"] = box
            info.pop("constraint_head", None)
            info["constraint_n_disjuncts"] = len(disj) if disj is not None else None
            summary["runs"][stem] = {k: v for k, v in info.items() if k != "constraint"}
            print(stem, info["soundness"], info["termination"], info["time_s"], "s,",
                  "check", chk["n_points"], "pts,", chk["n_mismatch"], "mismatch")
            summ_path.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
        if res_txt:
            (OUT_DIR / f"{name}.res").write_text("\n\n".join(res_txt), encoding="utf-8")


if __name__ == "__main__":
    main()
