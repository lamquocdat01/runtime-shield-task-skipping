"""c7a_check.py -- checklist C7(a): anonymous re-fetch and re-hash of the released repository.

  --make-manifest <repo>   write <repo>/MANIFEST.txt: "<sha256>  <path>" for every file tracked
                           by git (except MANIFEST.txt itself), from the bytes in the work tree
                           (.gitattributes has `* -text`, so these are the bytes GitHub serves)
  --check <tag>            download MANIFEST.txt and every listed file from
                           raw.githubusercontent.com/<owner>/<repo>/<tag>/<path> with plain
                           `requests` (no credentials, no token, no cookies), re-hash, compare;
                           writes a Markdown report. PASS only if 100 % of the files match.
Usage:
  python code/c7a_check.py --make-manifest .
  python code/c7a_check.py --check v1.2.0-submission --report C7A_CHECK.md
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from pathlib import Path

OWNER, REPO = "lamquocdat01", "runtime-shield-task-skipping"


def make_manifest(repo: Path) -> int:
    files = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True,
                           check=True).stdout.split("\n")
    files = sorted(f for f in files if f and f != "MANIFEST.txt")
    lines = [f"{hashlib.sha256((repo / f).read_bytes()).hexdigest()}  {f}" for f in files]
    (repo / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return len(lines)


def check(tag: str, report: Path) -> bool:
    import requests
    base = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{tag}/"
    s = requests.Session()
    s.trust_env = False                      # ignore netrc / proxy credentials
    r = s.get(base + "MANIFEST.txt", timeout=60)
    rows, ok_all = [], r.status_code == 200
    if ok_all:
        man = [ln.split("  ", 1) for ln in r.text.splitlines() if ln.strip()]
        for h, path in man:
            q = s.get(base + path, timeout=120)
            got = hashlib.sha256(q.content).hexdigest() if q.status_code == 200 else None
            rows.append((path, q.status_code, h, got, got == h))
            time.sleep(0.05)
        ok_all = all(x[4] for x in rows)
    lines = [f"# C7(a) — anonymous fetch + re-hash, tag `{tag}`", "",
             f"- Base URL: `{base}` (python requests, no credentials) · {time.strftime('%Y-%m-%d %H:%M')}",
             f"- MANIFEST.txt HTTP {r.status_code}",
             f"- Files: {len(rows)}; matching: {sum(x[4] for x in rows)}; **{'PASS' if ok_all else 'FAIL'}**",
             "", "| path | HTTP | match |", "|---|---|---|"]
    lines += [f"| `{p}` | {c} | {'yes' if m else 'NO'} |" for p, c, _, _, m in rows]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:5]))
    return ok_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="unused; recorded")
    ap.add_argument("--make-manifest", default="")
    ap.add_argument("--check", default="")
    ap.add_argument("--report", default="C7A_CHECK.md")
    a = ap.parse_args()
    if a.make_manifest:
        print(make_manifest(Path(a.make_manifest)), "files in MANIFEST.txt")
    if a.check:
        raise SystemExit(0 if check(a.check, Path(a.report)) else 1)


if __name__ == "__main__":
    main()
