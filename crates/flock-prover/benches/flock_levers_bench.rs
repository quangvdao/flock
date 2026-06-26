//! Flock-only benchmark levers vs the shared headline protocol.
//!
//! Measures setup prewarm, CSC warm, and scratch-pool modes while holding fixed
//! the conventions competitors also get: one untimed warmup prove and best-of-3
//! minimum timed `prove_fast` (matching `blake3_proof.rs`).
//!
//! ```sh
//! cargo build --release -p flock-prover --bench flock_levers_bench
//! FLOCK_LEVER=HEADLINE MEMORY_HASH=blake3 MEMORY_LOG2=18 \
//!   ./target/release/deps/flock_levers_bench-*
//! ```
//!
//! Environment:
//! - `FLOCK_LEVER` — `HEADLINE`, `NO_PREWARM`, `NO_CSC`, `NO_AB`, `POOL_SOFT`,
//!   `POOL_HARD`, or `ALL`
//! - `MEMORY_HASH` — `blake3` (default), `sha2`, or `keccak3`
//! - `MEMORY_N` / `MEMORY_LOG2` — batch size (default log2=18 for blake3)
//! - `FLOCK_INNER` — timed runs per outer repeat (default `3`, headline best-of-3)
//! - `FLOCK_OUTER` — outer repeats per config (default `5`)

use std::hint::black_box;
use std::time::Instant;

use flock_core::scratch;
use flock_prover::challenger::FsChallenger;
use flock_prover::r1cs_hashes::blake3::{Blake3Setup, Compression, K_LOG as BLAKE3_K_LOG};
use flock_prover::r1cs_hashes::common::SetupWarmOpts;
use flock_prover::r1cs_hashes::keccak::{STATE_BITS, State};
use flock_prover::r1cs_hashes::keccak3::KeccakSetup;
use flock_prover::r1cs_hashes::sha2::{K_LOG as SHA2_K_LOG, Sha256HybridSetup};
use flock_prover::r1cs_hashes::{blake3, keccak3, sha2};

/// How the global scratch pool is treated during timed proves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PoolMode {
    /// Production: reuse pooled buffers across timed runs (warmup fills the pool).
    On,
    /// Clear once after warmup, before timed runs. Competitor-like: same process,
    /// warmup-heated allocator, but no dedicated pool reuse from warmup.
    SoftClear,
    /// Clear before every timed run. Upper bound on dedicated-pool benefit.
    HardClear,
}

impl PoolMode {
    fn label(self) -> &'static str {
        match self {
            Self::On => "on",
            Self::SoftClear => "soft_clear",
            Self::HardClear => "hard_clear",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct LeverConfig {
    label: &'static str,
    warm: SetupWarmOpts,
    pool: PoolMode,
}

impl LeverConfig {
    const HEADLINE: Self = Self {
        label: "HEADLINE",
        warm: SetupWarmOpts::PRODUCTION,
        pool: PoolMode::On,
    };
    const NO_PREWARM: Self = Self {
        label: "NO_PREWARM",
        warm: SetupWarmOpts {
            csc_circuit: true,
            scratch_prewarm: false,
        },
        pool: PoolMode::On,
    };
    const NO_CSC: Self = Self {
        label: "NO_CSC",
        warm: SetupWarmOpts {
            csc_circuit: false,
            scratch_prewarm: true,
        },
        pool: PoolMode::On,
    };
    const NO_AB: Self = Self {
        label: "NO_AB",
        warm: SetupWarmOpts::COLD,
        pool: PoolMode::On,
    };
    const POOL_SOFT: Self = Self {
        label: "POOL_SOFT",
        warm: SetupWarmOpts::PRODUCTION,
        pool: PoolMode::SoftClear,
    };
    const POOL_HARD: Self = Self {
        label: "POOL_HARD",
        warm: SetupWarmOpts::PRODUCTION,
        pool: PoolMode::HardClear,
    };

    fn all() -> &'static [Self] {
        &[
            Self::HEADLINE,
            Self::NO_PREWARM,
            Self::NO_CSC,
            Self::NO_AB,
            Self::POOL_SOFT,
            Self::POOL_HARD,
        ]
    }

    fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_uppercase().as_str() {
            "HEADLINE" => Some(Self::HEADLINE),
            "NO_PREWARM" | "NO-PREWARM" | "MINUS_A" | "-A" => Some(Self::NO_PREWARM),
            "NO_CSC" | "NO-CSC" | "MINUS_B" | "-B" => Some(Self::NO_CSC),
            "NO_AB" | "NO-AB" | "MINUS_AB" | "-AB" => Some(Self::NO_AB),
            "POOL_SOFT" | "POOL-SOFT" | "MINUS_C_SOFT" | "-C-SOFT" => Some(Self::POOL_SOFT),
            "POOL_HARD" | "POOL-HARD" | "MINUS_C_HARD" | "-C-HARD" => Some(Self::POOL_HARD),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TrialStats {
    min_s: f64,
    median_s: f64,
    p95_s: f64,
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        (z ^ (z >> 31)) as u32
    }
    fn next_u64(&mut self) -> u64 {
        self.next_u32() as u64 | ((self.next_u32() as u64) << 32)
    }
}

fn random_blake3_block(rng: &mut Rng) -> Compression {
    let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    (cv, m, rng.next_u32() as u64, 64u32, 11u32)
}

fn random_sha2_block(rng: &mut Rng) -> ([u32; 8], [u32; 16]) {
    (
        std::array::from_fn(|_| rng.next_u32()),
        std::array::from_fn(|_| rng.next_u32()),
    )
}

fn random_keccak_state(rng: &mut Rng) -> State {
    let mut s = [false; STATE_BITS];
    let mut i = 0;
    while i < STATE_BITS {
        let w = rng.next_u64();
        for b in 0..64 {
            if i + b < STATE_BITS {
                s[i + b] = (w >> b) & 1 == 1;
            }
        }
        i += 64;
    }
    s
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn summarize(mut samples: Vec<f64>) -> TrialStats {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    TrialStats {
        min_s: samples[0],
        median_s: percentile(&samples, 0.5),
        p95_s: percentile(&samples, 0.95),
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn resolve_n(hash: &str, log2_default: usize) -> (usize, usize) {
    if let Ok(n) = std::env::var("MEMORY_N") {
        let n: usize = n.parse().expect("MEMORY_N must be a positive integer");
        assert!(n >= 1, "MEMORY_N must be ≥ 1");
        let m = match hash {
            "blake3" => BLAKE3_K_LOG + blake3::min_n_blocks_log(n),
            "sha2" => SHA2_K_LOG + sha2::min_n_blocks_log(n),
            "keccak3" => keccak3::K_LOG + keccak3::min_n_blocks_log(n),
            other => panic!("unknown MEMORY_HASH {other:?}"),
        };
        return (n, m);
    }
    let h = env_usize("MEMORY_LOG2", log2_default);
    let n = 1usize << h;
    let m = match hash {
        "blake3" => BLAKE3_K_LOG + h,
        "sha2" => SHA2_K_LOG + h,
        "keccak3" => keccak3::K_LOG + h,
        other => panic!("unknown MEMORY_HASH {other:?}"),
    };
    (n, m)
}

fn print_csv_header() {
    println!(
        "hash,n,m,config,prewarm,csc,pool_mode,outer,inner,min_ms,median_ms,p95_ms,throughput_per_s"
    );
}

fn print_csv_row(
    hash: &str,
    n: usize,
    m: usize,
    cfg: LeverConfig,
    outer: usize,
    inner: usize,
    stats: TrialStats,
) {
    let throughput = n as f64 / stats.min_s;
    println!(
        "{hash},{n},{m},{},{},{},{},{outer},{inner},{:.3},{:.3},{:.3},{:.0}",
        cfg.label,
        u8::from(cfg.warm.scratch_prewarm),
        u8::from(cfg.warm.csc_circuit),
        cfg.pool.label(),
        stats.min_s * 1000.0,
        stats.median_s * 1000.0,
        stats.p95_s * 1000.0,
        throughput,
    );
}

/// One outer repeat: setup → warmup → optional soft pool clear → best-of-N timed proves.
fn bench_blake3_outer(
    n: usize,
    cfg: LeverConfig,
    inner: usize,
    outer_seed: u64,
) -> TrialStats {
    let mk_blocks = |seed: u64| {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| random_blake3_block(&mut rng))
            .collect::<Vec<Compression>>()
    };
    let inputs: Vec<Vec<Compression>> = (0..=inner)
        .map(|t| mk_blocks(outer_seed ^ (t as u64)))
        .collect();

    scratch::clear();
    let setup = Blake3Setup::new_with_warm_opts(n, cfg.warm);

    {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let (p, _, _) = setup.prove_fast(&inputs[0], &mut ch);
        black_box(&p);
    }

    if cfg.pool == PoolMode::SoftClear {
        scratch::clear();
    }

    let mut samples = Vec::with_capacity(inner);
    for t in 0..inner {
        if cfg.pool == PoolMode::HardClear {
            scratch::clear();
        }
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let t0 = Instant::now();
        let (p, _, _) = setup.prove_fast(&inputs[t + 1], &mut ch);
        samples.push(t0.elapsed().as_secs_f64());
        black_box(&p);
    }
    summarize(samples)
}

fn bench_sha2_outer(n: usize, cfg: LeverConfig, inner: usize, outer_seed: u64) -> TrialStats {
    let mk_blocks = |seed: u64| {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| random_sha2_block(&mut rng))
            .collect::<Vec<([u32; 8], [u32; 16])>>()
    };
    let inputs: Vec<Vec<([u32; 8], [u32; 16])>> = (0..=inner)
        .map(|t| mk_blocks(outer_seed ^ (t as u64)))
        .collect();

    scratch::clear();
    let setup = Sha256HybridSetup::new_with_warm_opts(n, cfg.warm);

    {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let (p, _, _) = setup.prove_fast(&inputs[0], &mut ch);
        black_box(&p);
    }

    if cfg.pool == PoolMode::SoftClear {
        scratch::clear();
    }

    let mut samples = Vec::with_capacity(inner);
    for t in 0..inner {
        if cfg.pool == PoolMode::HardClear {
            scratch::clear();
        }
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let t0 = Instant::now();
        let (p, _, _) = setup.prove_fast(&inputs[t + 1], &mut ch);
        samples.push(t0.elapsed().as_secs_f64());
        black_box(&p);
    }
    summarize(samples)
}

fn bench_keccak3_outer(n: usize, cfg: LeverConfig, inner: usize, outer_seed: u64) -> TrialStats {
    let mk_states = |seed: u64| {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| random_keccak_state(&mut rng))
            .collect::<Vec<State>>()
    };
    let inputs: Vec<Vec<State>> = (0..=inner)
        .map(|t| mk_states(outer_seed ^ (t as u64)))
        .collect();

    scratch::clear();
    let setup = KeccakSetup::new_with_warm_opts(n, cfg.warm);

    {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let (p, _, _) = setup.prove_fast(&inputs[0], &mut ch);
        black_box(&p);
    }

    if cfg.pool == PoolMode::SoftClear {
        scratch::clear();
    }

    let mut samples = Vec::with_capacity(inner);
    for t in 0..inner {
        if cfg.pool == PoolMode::HardClear {
            scratch::clear();
        }
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let t0 = Instant::now();
        let (p, _, _) = setup.prove_fast(&inputs[t + 1], &mut ch);
        samples.push(t0.elapsed().as_secs_f64());
        black_box(&p);
    }
    summarize(samples)
}

fn bench_outer(hash: &str, n: usize, cfg: LeverConfig, inner: usize, outer_seed: u64) -> TrialStats {
    match hash {
        "blake3" => bench_blake3_outer(n, cfg, inner, outer_seed),
        "sha2" => bench_sha2_outer(n, cfg, inner, outer_seed),
        "keccak3" => bench_keccak3_outer(n, cfg, inner, outer_seed),
        other => panic!("unknown MEMORY_HASH {other:?}"),
    }
}

fn run_config(hash: &str, n: usize, m: usize, cfg: LeverConfig, inner: usize, outer: usize) {
    for o in 0..outer {
        let seed = 0xC0FFEE0000 ^ ((n as u64) << 16) ^ (o as u64);
        let stats = bench_outer(hash, n, cfg, inner, seed);
        print_csv_row(hash, n, m, cfg, o, inner, stats);
    }
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let hash = std::env::var("MEMORY_HASH").unwrap_or_else(|_| "blake3".into());
    let (n, m) = resolve_n(&hash, 18);
    let inner = env_usize("FLOCK_INNER", 3);
    let outer = env_usize("FLOCK_OUTER", 5);
    let lever_var = std::env::var("FLOCK_LEVER").unwrap_or_else(|_| "ALL".into());

    eprintln!(
        "flock_levers_bench: hash={hash} n={n} m={m} inner={inner} outer={outer} lever={lever_var}"
    );

    print_csv_header();

    if lever_var.eq_ignore_ascii_case("all") {
        for cfg in LeverConfig::all() {
            run_config(&hash, n, m, *cfg, inner, outer);
        }
        return;
    }

    let cfg = LeverConfig::parse(&lever_var)
        .unwrap_or_else(|| panic!("unknown FLOCK_LEVER {lever_var:?}"));
    run_config(&hash, n, m, cfg, inner, outer);
}
