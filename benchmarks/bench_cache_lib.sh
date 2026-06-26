# Shared benchmark result-cache helpers.
# Source from orchestrator scripts after setting FLOCK_ROOT (repo root).

# Short git SHA for the Flock tree; falls back when not a git checkout.
bench_cache_repo_stamp() {
	local root="${1:?}"
	git -C "$root" rev-parse --short=12 HEAD 2>/dev/null || echo "nogit"
}

# Cache path: <dir>/<prover>_2^<h>_t<threads>_<gitsha>
bench_cache_file() {
	local dir="$1" prover="$2" log2_h="$3" threads="$4" stamp="$5"
	echo "$dir/${prover}_2^${log2_h}_t${threads}_${stamp}"
}
