"""make_numbers.py -- C6/C7: every number the paper may print, extracted programmatically from
the result files into results/numbers.json (flat keys "<file>:<path>"). Numeric and boolean
leaves only. Keys whose path names a wall-clock / timing quantity are put in "volatile"
(machine-dependent, not compared by the reproduction check); all others are "stable".
`--check <numbers.json>` compares a freshly generated set key by key with a reference file and
writes REPRO_CHECK.md next to it.
Usage: python code/make_numbers.py [--check ref_numbers.json] [--report REPRO_CHECK.md]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

TOPIC_DIR = Path(__file__).resolve().parent.parent
RES = TOPIC_DIR / "results"
SOURCES = [
    "results/frontier_correct_info.json", "results/p1_t4.json", "results/p1_shield.json",
    "results/p1_conformance.json", "results/region_ext/bruteforce_summary.json",
    "results/region_ext/imitator_summary.json", "results/region_ext/cb_exact.json",
    "results/region_ext/cb_exact_KP_rule.json", "results/region_ext/cb_rule_ext.json",
    "results/region_ext/imitator_cb_final.json", "results/p2_t4.json", "results/p2_shield.json",
    "results/fi_baserate.json", "results/shield_size.json", "results/seeds5_summary.json",
]
VOLATILE = re.compile(r"(wall|seconds|time_s|_secs|_ms\b|ns_per_frame|U_k2|elapsed|rss|ratio_shield_to_detector)", re.I)
# Sole exception to the stable comparison (stays in "stable" in numbers.json, but --check treats it like a
# wall_s key). The six IMITATOR breaker runs of the template encoding (CB_*_k1-1_ge/gt) produce no result
# in either run and are not used by the paper. Five exit 124 (900 s timeout) both times; CB_d_block_k1-1_gt
# exited 1 after 407 s in the reference run (cause not determined) and 124 in the reproduction.
CHECK_AS_VOLATILE = {"imitator_summary.json:runs.CB_d_block_k1-1_gt.returncode"}


def flatten(obj, prefix, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flatten(v, f"{prefix}[{i}]", out)
    elif isinstance(obj, bool) or obj is None:
        if isinstance(obj, bool):
            out[prefix] = obj
    elif isinstance(obj, (int, float)):
        out[prefix] = obj
    elif isinstance(obj, str):
        try:
            x = float(obj)
            if math.isfinite(x) and re.fullmatch(r"-?\d+(\.\d+)?(e-?\d+)?", obj.strip()):
                out[prefix] = x
        except ValueError:
            pass


def proof_numbers(log: Path) -> dict:
    out = {}
    if not log.exists():
        return out
    txt = log.read_text(encoding="utf-8", errors="replace")
    goals = re.findall(r"Proved (\d+) of (\d+) total goals", txt)
    for i, (a, b) in enumerate(goals):
        out[f"verifier.goal_block[{i}].proved"] = int(a)
        out[f"verifier.goal_block[{i}].total"] = int(b)
    for name, verdict in re.findall(r'\("([A-Za-z0-9_]+)",(Valid|Invalid)\)', txt):
        out[f"theorem.{name}.valid"] = verdict == "Valid"
    m = re.search(r"VERIFY_MS=(\d+)", txt)
    if m:
        out["verifier.VERIFY_MS"] = int(m.group(1))
    return out


def build() -> dict:
    stable, volatile, srcs = {}, {}, {}
    for rel in SOURCES:
        p = TOPIC_DIR / rel
        if not p.exists():
            continue
        srcs[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
        flat = {}
        flatten(json.loads(p.read_text(encoding="utf-8")), "", flat)
        for k, v in flat.items():
            key = f"{Path(rel).name}:{k}"
            (volatile if VOLATILE.search(k) else stable)[key] = v
    for k, v in proof_numbers(TOPIC_DIR / "monitor" / "proof" / "gate_shield_verify.log").items():
        (volatile if VOLATILE.search(k) else stable)[f"gate_shield_verify.log:{k}"] = v
    return {"stable": stable, "volatile": volatile, "sources_sha256": srcs}


def same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default="")
    ap.add_argument("--report", default="")
    ap.add_argument("--out", default=str(RES / "numbers.json"))
    a = ap.parse_args()
    cur = build()
    Path(a.out).write_text(json.dumps(cur, indent=1, sort_keys=True), encoding="utf-8")
    print(f"{len(cur['stable'])} stable + {len(cur['volatile'])} volatile numbers -> {a.out}")
    if not a.check:
        return
    ref = json.loads(Path(a.check).read_text(encoding="utf-8"))
    exc = {k: (ref["stable"].get(k), cur["stable"].get(k)) for k in sorted(CHECK_AS_VOLATILE)}
    rs = {k: v for k, v in ref["stable"].items() if k not in CHECK_AS_VOLATILE}
    cs = {k: v for k, v in cur["stable"].items() if k not in CHECK_AS_VOLATILE}
    missing = sorted(set(rs) - set(cs))
    extra = sorted(set(cs) - set(rs))
    diff = sorted(k for k in set(rs) & set(cs) if not same(rs[k], cs[k]))
    by_file = {}
    for k in rs:
        f = k.split(":")[0]
        d = by_file.setdefault(f, [0, 0, 0])
        d[0] += 1
        if k in cs and same(rs[k], cs[k]):
            d[1] += 1
        elif k not in cs:
            d[2] += 1
    lines = ["# REPRO_CHECK — numbers.json, key by key", "",
             f"Reference: `{a.check}` · regenerated: `{a.out}`", "",
             f"- stable keys in reference: **{len(ref['stable'])}**, of which compared: **{len(rs)}**; identical (rel. tol. 1e-9): "
             f"**{len(rs) - len(diff) - len(missing)}**; different: **{len(diff)}**; "
             f"missing in the rerun: **{len(missing)}**; new keys: {len(extra)}",
             f"- volatile (timing) keys, not compared: {len(ref.get('volatile', {}))}",
             f"- stable keys compared as volatile (sole exception, see below): {len(exc)}", "",
             "| Source file | keys | identical | missing |", "|---|---|---|---|"]
    lines += [f"| {f} | {n} | {ok} | {mi} |" for f, (n, ok, mi) in sorted(by_file.items())]
    if diff:
        lines += ["", "## Different keys (first 50)", "", "| key | reference | rerun |", "|---|---|---|"]
        lines += [f"| `{k}` | {rs[k]} | {cs[k]} |" for k in diff[:50]]
    if missing:
        lines += ["", "## Missing keys (first 50)", ""] + [f"- `{k}`" for k in missing[:50]]
    lines += ["", "## Sole exception: exit code of a run without result", "",
              "| key | reference | rerun |", "|---|---|---|"]
    lines += [f"| `{k}` | {r} | {c} |" for k, (r, c) in exc.items()]
    lines += ["", "The six IMITATOR breaker runs of the template encoding (`CB_k1-1_ge/gt`, `CB_d_pipe_k1-1_ge/gt`, "
              "`CB_d_block_k1-1_ge/gt`, from `code/region_imitator.py`) produce no result in either run "
              "(`integer_check.n_points` = 0) and are not used by the paper; its breaker checks use the "
              "discrete encoding of `code/imitator_cb_final.py`. Five exit with code 124 (900 s timeout) in both "
              "runs. `CB_d_block_k1-1_gt` exited with code 1 after 407 s in the reference run (cause not "
              "determined) and with 124 at the timeout in the reproduction. Its `returncode` is therefore "
              "compared like the volatile `wall_s` keys; this is the only stable key so treated."]
    rep = Path(a.report) if a.report else Path(a.out).with_name("REPRO_CHECK.md")
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
