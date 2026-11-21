#!/bin/bash
# Static size of the generated shield C (no hardware needed): gcc -c at -O2 and -Os, then `size -A`.
# Host ISA (x86-64 under WSL2); not a measurement on an embedded target.
# Usage (from Git Bash): MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -- bash /mnt/<repo>/monitor/size_shield.sh
set -eu
SRC="$(cd "$(dirname "$0")" && pwd)"
OUT="$SRC/../results/shield_size.json"
cd "$SRC"
GCC="$(gcc --version | head -1)"
row() {
  local O=$1; gcc -$O -std=c99 -c gate_shield.c -o /tmp/gate_shield_$O.o
  local t d b
  t=$(size -A /tmp/gate_shield_$O.o | awk '$1==".text"{print $2}')
  d=$(size -A /tmp/gate_shield_$O.o | awk '$1==".data"{print $2}')
  b=$(size -A /tmp/gate_shield_$O.o | awk '$1==".bss"{print $2}')
  echo "\"$O\": {\"text_bytes\": $t, \"data_bytes\": $d, \"bss_bytes\": $b, \"state_bytes\": $((d + b))}"
}
{
  echo "{\"compiler\": \"$GCC\", \"isa\": \"x86-64 (host)\", \"source\": \"monitor/gate_shield.c\","
  echo " \"O2\": {$(row O2 | cut -d'{' -f2-),"
  echo " \"Os\": {$(row Os | cut -d'{' -f2-)}"
} > "$OUT"
cat "$OUT"
