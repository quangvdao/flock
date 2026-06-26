#!/usr/bin/env bash
# Run memory-regime benchmarks and write CSV results with environment metadata.
#
# Usage (from repo root):
#   ./benchmarks/run_memory_regime.sh
#   MEMORY_HASH=sha2 MEMORY_LOG2=17 FLOCK_TRIALS=10 ./benchmarks/run_memory_regime.sh
#   ./benchmarks/run_memory_regime.sh --r3-subprocess   # true cold-process R3
#
# Output: benchmarks/memory-regime-results/<timestamp>_<hash>_log2<h>.csv

set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$BASE/.." && pwd)"
OUT_DIR="$BASE/memory-regime-results"
mkdir -p "$OUT_DIR"

HASH="${MEMORY_HASH:-blake3}"
LOG2="${MEMORY_LOG2:-18}"
TRIALS="${FLOCK_TRIALS:-10}"
DO_R3_SUB=0

for arg in "$@"; do
	case "$arg" in
		--r3-subprocess) DO_R3_SUB=1 ;;
		-h|--help)
			sed -n '2,12p' "$0"
			exit 0
			;;
		*) echo "unknown arg: $arg" >&2; exit 1 ;;
	esac
done

cd "$ROOT"
cargo build --release -p flock-prover --bench memory_regime_bench --quiet
BIN="$(ls target/release/deps/memory_regime_bench-* 2>/dev/null | head -1)"
[[ -n "$BIN" && -x "$BIN" ]] || { echo "memory_regime_bench binary not found" >&2; exit 1; }

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$OUT_DIR/${STAMP}_${HASH}_log2${LOG2}.csv"
META="$OUT_DIR/${STAMP}_${HASH}_log2${LOG2}.meta"

{
	echo "# memory_regime run metadata"
	echo "timestamp_utc=$STAMP"
	echo "git_sha=$(git rev-parse HEAD)"
	echo "git_describe=$(git describe --always --dirty 2>/dev/null || true)"
	echo "rustc=$(rustc --version 2>/dev/null || echo n/a)"
	echo "host=$(uname -a)"
	echo "memory_hash=$HASH"
	echo "memory_log2=$LOG2"
	echo "flock_trials=$TRIALS"
	echo "rayon_num_threads=${RAYON_NUM_THREADS:-default}"
} >"$META"

echo "Building done. Running in-process regimes (PREWARM,R0,R1,R2,R4)..." >&2
MEMORY_HASH="$HASH" MEMORY_LOG2="$LOG2" FLOCK_TRIALS="$TRIALS" FLOCK_REGIME=ALL \
	"$BIN" 2>"$OUT_DIR/${STAMP}.log" | tee "$OUT"

if [[ "$DO_R3_SUB" == 1 ]]; then
	echo "Running R3 subprocess trials ($TRIALS fresh processes)..." >&2
	echo "" >>"$OUT"
	echo "# R3 subprocess (one prove per fresh process, production setup, no warmup)" >>"$OUT"
	# Header already printed; append R3 rows only.
	r3_samples=()
	for _ in $(seq 1 "$TRIALS"); do
		line="$(MEMORY_HASH="$HASH" MEMORY_LOG2="$LOG2" FLOCK_TRIALS=1 FLOCK_REGIME=R3 FLOCK_WARMUP=0 \
			"$BIN" 2>/dev/null | tail -1)"
		r3_samples+=("$line")
		echo "$line" >>"$OUT"
	done
fi

echo "Wrote $OUT" >&2
echo "Meta: $META" >&2
