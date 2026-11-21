# Toolchain

Host used: Windows 11, Intel i7-1185G7, 32 GB; Python 3.11.9 (Windows) with `requirements.txt`.
Formal tools run in WSL2 `Ubuntu-24.04` under `/opt/vgc`:

| Tool | Version | Install |
|---|---|---|
| IMITATOR | 3.4 "Cheese Durian" (static binary v3.4.0, sha256 `3d1d4295…cc73`) | `curl -sSLO https://github.com/imitator-model-checker/imitator/releases/download/v3.4.0/imitator-v3.4.0-linux-amd64` → `/usr/local/bin/imitator` |
| GHC / cabal | 9.6.7 / 3.16.1.0 (ghcup 0.2.6.2) | ghcup bootstrap with `BOOTSTRAP_HASKELL_GHC_VERSION=9.6.7` |
| Copilot, copilot-c99, copilot-theorem, copilot-verifier | 4.8.1 (Hackage) | `monitor/gateshield.cabal` |
| crucible / crucible-llvm / crux-llvm / what4 | 0.7.2 / 0.7.1 / 0.9 / 1.7.3 | resolved by cabal |
| clang / LLVM | 14.0.6 (`apt-get install clang-14 llvm-14`, unversioned links first in `PATH`) | copilot-verifier needs LLVM <= 16 |
| Z3 | 4.8.12 (Ubuntu package) | |
| gcc | 13.3.0 | shield shared objects and benchmark |
| UPPAAL | 5.0.0 (verifyta) | needs an academic licence key (`verifyta --key <KEY> --lease 168`) |

Run from Git Bash with `MSYS_NO_PATHCONV=1` (otherwise `/opt/...` paths are rewritten), and call
script files (`wsl.exe -d Ubuntu-24.04 -u root -- bash <script>`): `bash -lc "<cmd>"` re-parses
`$VAR` before WSL sees it.

- Copilot build + proofs + verifier: `bash monitor/build_shield.sh` (writes `monitor/proof/gate_shield_verify.log`).
- IMITATOR runs: `code/region_imitator.py` (P1 grid), `code/imitator_cb_final.py` (circuit breaker).
- Shield on the corpus + overhead: `code/shield_bridge.py` (needs `python3-numpy` in WSL).
