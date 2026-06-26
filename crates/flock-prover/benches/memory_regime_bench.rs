//! Memory-regime benchmark: quantify prewarm, scratch-pool reuse, and setup cost
//! relative to headline `prove_fast` timings.
//!
//! Run (from repo root):
//! ```sh
//! cargo build --release -p flock-prover --bench memory_regime_bench
//! MEMORY_HASH=blake3 MEMORY_LOG2=18 FLOCK_REGIME=ALL FLOCK_TRIALS=10 \
//!   ./target/release/deps/memory_regime_bench-* 2>/dev/null | head
//! ```
//!
//! Environment:
//! - `MEMORY_HASH` — `blake3` (default), `sha2`, or `keccak3`
//! - `MEMORY_N` — instance count (overrides `MEMORY_LOG2` when set)
//! - `MEMORY_LOG2` — `log2` instance count (default `18` for blake3/sha2)
//! - `FLOCK_REGIME` — `PREWARM`, `R0`, `R1`, `R2`, `R3`, `R4`, or `ALL`
//! - `FLOCK_TRIALS` — timed trials per regime (default `10`)
//! - `FLOCK_WARMUP` — `1`/`0` untimed warmup prove before trials (default `1` for R0/R1)
//! - `FLOCK_SCRATCH_CLEAR` — `1`/`0` call `scratch::clear()` before each trial (default per regime)

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Regime {
    Prewarm,
    R0,
    R1,
    R2,
    R3,
    R4,
}

impl Regime {
    fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "prewarm" => Some(Self::Prewarm),
            "r0" => Some(Self::R0),
            "r1" => Some(Self::R1),
            "r2" => Some(Self::R2),
            "r3" => Some(Self::R3),
            "r4" => Some(Self::R4),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Prewarm => "PREWARM",
            Self::R0 => "R0",
            Self::R1 => "R1",
            Self::R2 => "R2",
            Self::R3 => "R3",
            Self::R4 => "R4",
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

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name).ok().as_deref() {
        Some("1") | Some("true") | Some("TRUE") | Some("yes") => true,
        Some("0") | Some("false") | Some("FALSE") | Some("no") => false,
        Some(other) => panic!("{name}: expected 0/1, got {other:?}"),
        None => default,
    }
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

fn regime_warm(regime: Regime) -> SetupWarmOpts {
    match regime {
        Regime::R1 => SetupWarmOpts::COLD,
        Regime::R0 | Regime::R2 | Regime::R3 | Regime::R4 => SetupWarmOpts::PRODUCTION,
        Regime::Prewarm => SetupWarmOpts::COLD,
    }
}

fn regime_warmup(regime: Regime) -> bool {
    match regime {
        Regime::R2 | Regime::R3 => false,
        Regime::Prewarm => false,
        Regime::R0 | Regime::R1 | Regime::R4 => env_bool("FLOCK_WARMUP", true),
    }
}

fn regime_scratch_clear(regime: Regime) -> bool {
    match regime {
        Regime::R2 => true,
        Regime::R0 | Regime::R1 | Regime::R3 | Regime::R4 | Regime::Prewarm => {
            env_bool("FLOCK_SCRATCH_CLEAR", false)
        }
    }
}

fn print_csv_header() {
    println!(
        "hash,n,m,regime,trials,min_ms,median_ms,p95_ms,throughput_per_s,setup_ms,prewarm_ms"
    );
}

fn print_csv_row(
    hash: &str,
    n: usize,
    m: usize,
    regime: Regime,
    trials: usize,
    stats: TrialStats,
    setup_ms: Option<f64>,
    prewarm_ms: Option<f64>,
) {
    let throughput = n as f64 / stats.min_s;
    println!(
        "{hash},{n},{m},{},{trials},{:.3},{:.3},{:.3},{:.0},{},{}",
        regime.label(),
        stats.min_s * 1000.0,
        stats.median_s * 1000.0,
        stats.p95_s * 1000.0,
        throughput,
        setup_ms
            .map(|x| format!("{x:.3}"))
            .unwrap_or_else(|| "-".into()),
        prewarm_ms
            .map(|x| format!("{x:.3}"))
            .unwrap_or_else(|| "-".into()),
    );
}

fn bench_prewarm_only(m: usize, trials: usize) -> f64 {
    let mut samples = Vec::with_capacity(trials);
    for t in 0..trials {
        scratch::clear();
        let t0 = Instant::now();
        scratch::prewarm_prover(m);
        black_box(m.wrapping_add(t));
        samples.push(t0.elapsed().as_secs_f64());
    }
    summarize(samples).median_s * 1000.0
}

fn bench_blake3(n: usize, m: usize, regime: Regime, trials: usize) -> (TrialStats, Option<f64>, Option<f64>) {
    if regime == Regime::Prewarm {
        let ms = bench_prewarm_only(m, trials);
        return (
            TrialStats {
                min_s: f64::NAN,
                median_s: f64::NAN,
                p95_s: f64::NAN,
            },
            None,
            Some(ms),
        );
    }

    let warm = regime_warm(regime);
    let do_warmup = regime_warmup(regime);
    let clear_each = regime_scratch_clear(regime);

    let mk_blocks = |seed: u64| {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| random_blake3_block(&mut rng))
            .collect::<Vec<Compression>>()
    };
    let inputs: Vec<Vec<Compression>> = (0..=trials)
        .map(|t| mk_blocks(0xBEEF ^ (n as u64) ^ (t as u64)))
        .collect();

    let setup_ms = if regime == Regime::R4 {
        scratch::clear();
        let t0 = Instant::now();
        let setup = Blake3Setup::new_with_warm_opts(n, warm);
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let (p, _, _) = setup.prove_fast(&inputs[0], &mut ch);
        black_box(&p);
        Some(t0.elapsed().as_secs_f64() * 1000.0)
    } else {
        None
    };

    scratch::clear();
    let setup = Blake3Setup::new_with_warm_opts(n, warm);

    if do_warmup {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let (p, _, _) = setup.prove_fast(&inputs[0], &mut ch);
        black_box(&p);
    }

    let mut samples = Vec::with_capacity(trials);
    for t in 0..trials {
        if clear_each {
            scratch::clear();
        }
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let t0 = Instant::now();
        let (p, _, _) = setup.prove_fast(&inputs[t + 1], &mut ch);
        samples.push(t0.elapsed().as_secs_f64());
        black_box(&p);
    }

    (summarize(samples), setup_ms, None)
}

fn bench_sha2(n: usize, m: usize, regime: Regime, trials: usize) -> (TrialStats, Option<f64>, Option<f64>) {
    if regime == Regime::Prewarm {
        let ms = bench_prewarm_only(m, trials);
        return (
            TrialStats {
                min_s: f64::NAN,
                median_s: f64::NAN,
                p95_s: f64::NAN,
            },
            None,
            Some(ms),
        );
    }

    let warm = regime_warm(regime);
    let do_warmup = regime_warmup(regime);
    let clear_each = regime_scratch_clear(regime);

    let mk_blocks = |seed: u64| {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| random_sha2_block(&mut rng))
            .collect::<Vec<([u32; 8], [u32; 16])>>()
    };
    let inputs: Vec<Vec<([u32; 8], [u32; 16])>> = (0..=trials)
        .map(|t| mk_blocks(0xBEEF ^ (n as u64) ^ (t as u64)))
        .collect();

    let setup_ms = if regime == Regime::R4 {
        scratch::clear();
        let t0 = Instant::now();
        let setup = Sha256HybridSetup::new_with_warm_opts(n, warm);
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let (p, _, _) = setup.prove_fast(&inputs[0], &mut ch);
        black_box(&p);
        Some(t0.elapsed().as_secs_f64() * 1000.0)
    } else {
        None
    };

    scratch::clear();
    let setup = Sha256HybridSetup::new_with_warm_opts(n, warm);

    if do_warmup {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let (p, _, _) = setup.prove_fast(&inputs[0], &mut ch);
        black_box(&p);
    }

    let mut samples = Vec::with_capacity(trials);
    for t in 0..trials {
        if clear_each {
            scratch::clear();
        }
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let t0 = Instant::now();
        let (p, _, _) = setup.prove_fast(&inputs[t + 1], &mut ch);
        samples.push(t0.elapsed().as_secs_f64());
        black_box(&p);
    }

    (summarize(samples), setup_ms, None)
}

fn bench_keccak3(n: usize, m: usize, regime: Regime, trials: usize) -> (TrialStats, Option<f64>, Option<f64>) {
    if regime == Regime::Prewarm {
        let ms = bench_prewarm_only(m, trials);
        return (
            TrialStats {
                min_s: f64::NAN,
                median_s: f64::NAN,
                p95_s: f64::NAN,
            },
            None,
            Some(ms),
        );
    }

    let warm = regime_warm(regime);
    let do_warmup = regime_warmup(regime);
    let clear_each = regime_scratch_clear(regime);

    let mk_states = |seed: u64| {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| random_keccak_state(&mut rng))
            .collect::<Vec<State>>()
    };
    let inputs: Vec<Vec<State>> = (0..=trials)
        .map(|t| mk_states(0xBEEF ^ (n as u64) ^ (t as u64)))
        .collect();

    let setup_ms = if regime == Regime::R4 {
        scratch::clear();
        let t0 = Instant::now();
        let setup = KeccakSetup::new_with_warm_opts(n, warm);
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let (p, _, _) = setup.prove_fast(&inputs[0], &mut ch);
        black_box(&p);
        Some(t0.elapsed().as_secs_f64() * 1000.0)
    } else {
        None
    };

    scratch::clear();
    let setup = KeccakSetup::new_with_warm_opts(n, warm);

    if do_warmup {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let (p, _, _) = setup.prove_fast(&inputs[0], &mut ch);
        black_box(&p);
    }

    let mut samples = Vec::with_capacity(trials);
    for t in 0..trials {
        if clear_each {
            scratch::clear();
        }
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let t0 = Instant::now();
        let (p, _, _) = setup.prove_fast(&inputs[t + 1], &mut ch);
        samples.push(t0.elapsed().as_secs_f64());
        black_box(&p);
    }

    (summarize(samples), setup_ms, None)
}

fn run_regime(hash: &str, n: usize, m: usize, regime: Regime, trials: usize) {
    let (stats, setup_ms, prewarm_ms) = match hash {
        "blake3" => bench_blake3(n, m, regime, trials),
        "sha2" => bench_sha2(n, m, regime, trials),
        "keccak3" => bench_keccak3(n, m, regime, trials),
        other => panic!("unknown MEMORY_HASH {other:?}"),
    };
    print_csv_row(hash, n, m, regime, trials, stats, setup_ms, prewarm_ms);
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let hash = std::env::var("MEMORY_HASH").unwrap_or_else(|_| "blake3".into());
    let (n, m) = resolve_n(&hash, 18);
    let trials = env_usize("FLOCK_TRIALS", 10);
    let regime_var = std::env::var("FLOCK_REGIME").unwrap_or_else(|_| "ALL".into());

    eprintln!(
        "memory_regime_bench: hash={hash} n={n} m={m} trials={trials} regime={regime_var}"
    );

    print_csv_header();

    if regime_var.eq_ignore_ascii_case("all") {
        for regime in [
            Regime::Prewarm,
            Regime::R0,
            Regime::R1,
            Regime::R2,
            Regime::R4,
        ] {
            run_regime(&hash, n, m, regime, trials);
        }
        return;
    }

    let regime = Regime::parse(&regime_var)
        .unwrap_or_else(|| panic!("unknown FLOCK_REGIME {regime_var:?}"));
    run_regime(&hash, n, m, regime, trials);
}
