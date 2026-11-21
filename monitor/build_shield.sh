#!/bin/bash
# Build GateShield.hs in WSL, generate C, prove props (copilot-theorem / Z3), run
# copilot-verifier (C == spec, memory safety). Usage (from Git Bash):
#   MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/d/.../monitor/build_shield.sh
set -u
SRC="$(cd "$(dirname "$0")" && pwd)"
source /opt/vgc/.ghcup/env
export PATH=/opt/vgc/llvm14bin:$PATH
W=/opt/vgc/shield
mkdir -p $W && cp "$SRC/GateShield.hs" "$SRC/gateshield.cabal" $W/ && cd $W && sed -i 's/\r$//' GateShield.hs gateshield.cabal
mkdir -p "$SRC/proof"
LOG="$SRC/proof/gate_shield_verify.log"
{
echo "== build $(date -Is)"
echo "ghc $(ghc --numeric-version); cabal $(cabal --numeric-version); $(clang --version | head -1); $(z3 --version)"
T0=$(date +%s%N)
cabal build -j2 gate-shield 2>&1 | tail -5
echo "BUILD_MS=$(( ($(date +%s%N) - T0) / 1000000 ))"
BIN=$(cabal list-bin gate-shield)
mkdir -p gen && cd gen
echo "== compile (C generation)"
$BIN compile && ls -l gate_shield.* gate_ctl.* && sha256sum gate_shield.c gate_shield.h gate_ctl.c gate_ctl.h
cd $W && mkdir -p prove && cd prove
echo "== copilot-theorem (What4 / Z3) properties"
T1=$(date +%s%N)
$BIN prove
echo "PROVE_MS=$(( ($(date +%s%N) - T1) / 1000000 ))"
cd $W && mkdir -p verify && cd verify
echo "== copilot-verifier"
T2=$(date +%s%N)
/usr/bin/time -v $BIN verify 2>&1 | grep -v '^\s*$' | grep -E 'copilot-verifier|Proved|Maximum resident|Elapsed|error|rror|FAIL|fail' 
echo "rc=${PIPESTATUS[0]} VERIFY_MS=$(( ($(date +%s%N) - T2) / 1000000 ))"
} 2>&1 | tee "$LOG"
cp $W/gen/gate_shield.c $W/gen/gate_shield.h $W/gen/gate_ctl.c $W/gen/gate_ctl.h "$SRC/" 2>/dev/null
[ -f $W/gen/gate_shield_types.h ] && cp $W/gen/gate_shield_types.h $W/gen/gate_ctl_types.h "$SRC/"
echo done
