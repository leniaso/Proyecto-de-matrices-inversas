#!/usr/bin/env bash
# Uso: SIZES="512 1024" REPS=30 N0=128 ./bench.sh
# Opcional: TASKSET="taskset -c 0-7"
set -euo pipefail
SIZES=${SIZES:-"512 1024 2048 4096"}
REPS=${REPS:-30}; REPS_BIG=${REPS_BIG:-10}; N0=${N0:-128}
OUT=${OUT:-results.csv}; TASKSET=${TASKSET:-}
NCPU=$(nproc)

./matinv --header > "$OUT"
$TASKSET ./matinv seq 512 1 $N0 1 > /dev/null          # calentamiento

jobs=()
for n in $SIZES; do
  for r in $(seq 1 $REPS); do
    [[ $n -ge 4096 && $r -gt $REPS_BIG ]] && continue
    jobs+=("seq $n 1 $r")
    for v in threads procs hier; do
      for P in 2 4 8; do
        [[ $P -gt $NCPU ]] && continue
        jobs+=("$v $n $P $r")
      done
    done
  done
done
echo "Corridas totales: ${#jobs[@]}" >&2

printf '%s\n' "${jobs[@]}" | shuf | while read -r v n P r; do   # orden aleatorizado
  if [[ $r -gt 1 ]]; then export NORES=1; else unset NORES; fi    # residuo solo en la réplica 1
  $TASKSET ./matinv "$v" "$n" "$P" "$N0" $((1000 + r)) >> "$OUT"
done
