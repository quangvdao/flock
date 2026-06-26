#!/usr/bin/env bash
# Run Flock-only benchmark levers (prewarm, CSC warm, scratch pool modes).
#
# Holds fixed the shared competitor protocol: 1 untimed warmup prove + best-of-3
# minimum prove_fast. Varies only Flock-specific setup/pool behavior.
#
# Usage (from repo root):
#   ./benchmarks/run_flock_only_levers.sh
#   MEMORY_HASH=sha2 MEMORY_LOG2=17 ./benchmarks/run_flock_only_levers.sh
#   FLOCK_LEVER=HEADLINE FLOCK_OUTER=3 ./benchmarks/run_flock_only_levers.sh
#
# Configs (FLOCK_LEVER=ALL runs all, with cooldown between each):
#   HEADLINE   — production prewarm + CSC + pool on
#   NO_PREWARM — no scratch prewarm at setup
#   NO_CSC     — no CSC lincheck warm at setup
#   NO_AB      — neither prewarm nor CSC
#   POOL_SOFT  — clear pool once after warmup (competitor-like allocator reuse)
#   POOL_HARD  — clear pool before each timed run (upper bound on pool benefit)
#
# Output: benchmarks/flock-levers-results/<timestamp>_<hash>.csv

set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$BASE/.." && pwd)"
OUT_DIR="$BASE/flock-levers-results"
mkdir -p "$OUT_DIR"

HASH="${MEMORY_HASH:-blake3}"
LOG2="${MEMORY_LOG2:-18}"
INNER="${FLOCK_INNER:-3}"
OUTER="${FLOCK_OUTER:-5}"
COOLDOWN="${COOLDOWN:-30}"
LEVER="${FLOCK_LEVER:-ALL}"

[[ "$COOLDOWN" =~ ^[0-9]+$ ]] || {
	echo "COOLDOWN must be a non-negative integer (seconds), got '$COOLDOWN'" >&2
	exit 1
}

cd "$ROOT"
cargo build --release -p flock-prover --bench flock_levers_bench --quiet
BIN="$(ls target/release/deps/flock_levers_bench-* 2>/dev/null | head -1)"
[[ -n "$BIN" && -x "$BIN" ]] || {
	echo "flock_levers_bench binary not found" >&2
	exit 1
}

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$OUT_DIR/${STAMP}_${HASH}_log2${LOG2}.csv"
META="$OUT_DIR/${STAMP}_${HASH}_log2${LOG2}.meta"

{
	echo "# flock-only levers run metadata"
	echo "timestamp_utc=$STAMP"
	echo "git_sha=$(git rev-parse HEAD)"
	echo "git_describe=$(git describe --always --dirty 2>/dev/null || true)"
	echo "rustc=$(rustc --version 2>/dev/null || echo n/a)"
	echo "host=$(uname -a)"
	echo "memory_hash=$HASH"
	echo "memory_log2=$LOG2"
	echo "flock_inner=$INNER"
	echo "flock_outer=$OUTER"
	echo "flock_lever=$LEVER"
	echo "cooldown_s=$COOLDOWN"
	echo "rayon_num_threads=${RAYON_NUM_THREADS:-default}"
} >"$META"

run_one_lever() {
	local lever="$1"
	echo "=== FLOCK_LEVER=$lever ($(date -u)) ===" >&2
	MEMORY_HASH="$HASH" MEMORY_LOG2="$LOG2" FLOCK_INNER="$INNER" FLOCK_OUTER="$OUTER" \
		FLOCK_LEVER="$lever" "$BIN" 2>>"$OUT_DIR/${STAMP}.log"
}

if [[ "$LEVER" == "ALL" || "$LEVER" == "all" ]]; then
	LEVERS=(HEADLINE NO_PREWARM NO_CSC NO_AB POOL_SOFT POOL_HARD)
	{
		echo "# flock-only levers: shared protocol = 1 warmup + best-of-${INNER} min; outer=${OUTER} repeats/config"
		echo "# POOL_SOFT = competitor-like (clear pool once after warmup)"
		echo "# POOL_HARD = upper bound (clear pool before each timed run)"
	} >"$OUT"
	first=1
	for lever in "${LEVERS[@]}"; do
		if (( first )); then
			first=0
		elif (( COOLDOWN > 0 )); then
			echo "  cooldown: sleeping ${COOLDOWN}s before ${lever}..." >&2
			sleep "$COOLDOWN"
		fi
		run_one_lever "$lever" >>"$OUT"
	done
else
	run_one_lever "$LEVER" | tee "$OUT"
fi

echo "Wrote $OUT" >&2
echo "Meta: $META" >&2
