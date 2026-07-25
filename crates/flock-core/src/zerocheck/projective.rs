//! Projective ("monomial-basis" / `{0,∞}`-interpolating-set) multilinear
//! sum-check representation for the prover's rolling tables.
//!
//! # Representation
//!
//! Flock's baseline stores a multilinear polynomial by its evaluations on the
//! Boolean hypercube. With the low bit bound first, a pair is
//! `(a(0,x), a(1,x))` and binding at `r` costs
//!
//! ```text
//!   a'[x] = a0 + r · (a1 + a0)          // 2 XOR + 1 mul   (char 2: + == −)
//! ```
//!
//! The projective variant instead stores the pair in the monomial basis of the
//! variable being bound: `(c0, c1)` with `a(X) = c0 + X · c1`, i.e.
//! `c0 = a(0)` and `c1 = a(0) + a(1) = a(∞)` (the leading coefficient). This is
//! the `{0,∞}` interpolating set. Binding then costs
//!
//! ```text
//!   a'[x] = c0 + r · c1                 // 1 XOR + 1 mul
//! ```
//!
//! # Why the basis change is self-sustaining, and where the win actually comes from
//!
//! Binding a twisted table yields the *evaluation* table of the remaining
//! variables, so the next round needs its lowest variable re-twisted — one XOR
//! per output pair. Naively that gives back most of what binding saved.
//!
//! The point is that the baseline **already pays exactly that XOR**: its round
//! message needs `a(∞) = a0 + a1` for the `g_inf` term (the Karatsuba ∞-trick
//! Flock already uses). So in a fused fold-plus-message kernel the re-twist is
//! free — it *is* the `g_inf` operand — and the projective form additionally
//! gets `a(1)` for free as the raw folded value. Per output pair per polynomial:
//!
//! | | fold | re-twist | message | total |
//! |---|---|---|---|---|
//! | Boolean  | 4 XOR + 2 mul | — | 1 XOR | **5 XOR + 2 mul** |
//! | Projective | 2 XOR + 2 mul | 1 XOR | 0 XOR | **3 XOR + 2 mul** |
//!
//! Multiply counts are **identical**; only F128 additions differ, 5 → 3. In
//! characteristic 2 an F128 addition is a single 128-bit `EOR`, whereas a
//! multiply is a PMULL chain, so the expected wall-clock gain is modest.

use crate::field::{F128, F256Unreduced};
use crate::zerocheck::univariate_skip::build_eq;

#[cfg(test)]
pub(crate) fn twist_lowest_in_place(a: &mut [F128]) {
    let n = a.len();
    assert!(n.is_multiple_of(2), "twist needs an even number of entries");
    let mut i = 0;
    while i < n {
        a[i + 1] = a[i] + a[i + 1];
        i += 2;
    }
}

/// Small-table in-place projective round used below the parallel threshold.
///
/// The write cursor stays behind the read cursor, so folding and re-twisting
/// can safely reuse the input allocations without scratch buffers.
pub(super) fn fold_and_compute_round_pair_in_place(
    a: &mut Vec<F128>,
    b: &mut Vec<F128>,
    r_fold: F128,
    r_next: &[F128],
) -> (F128, F128) {
    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n.is_power_of_two() && n >= 4);
    let log_n = n.trailing_zeros() as usize;
    assert_eq!(r_next.len(), log_n - 1);
    let eq = build_eq(&r_next[1..]);
    assert_eq!(eq.len(), n / 4);

    let mut p1 = F256Unreduced::ZERO;
    let mut pinf = F256Unreduced::ZERO;
    for (u, &eq_u) in eq.iter().enumerate() {
        let s = 4 * u;
        let (a0, a1) = fold_pair_projective(a, s, r_fold);
        let (b0, b1) = fold_pair_projective(b, s, r_fold);
        let a_inf = a0 + a1;
        let b_inf = b0 + b1;
        let o = 2 * u;
        a[o] = a0;
        a[o + 1] = a_inf;
        b[o] = b0;
        b[o + 1] = b_inf;
        p1 ^= eq_u.mul_unreduced(a1 * b1);
        pinf ^= eq_u.mul_unreduced(a_inf * b_inf);
    }
    a.truncate(n / 2);
    b.truncate(n / 2);
    (r_next[0] * p1.reduce(), pinf.reduce())
}

/// Fold two adjacent projective-basis pairs at `r`. The odd slots are already
/// the coefficients `c1`, so the two difference XORs are gone.
#[inline(always)]
fn fold_pair_projective(src: &[F128], s: usize, r: F128) -> (F128, F128) {
    let e0 = src[s];
    let c0 = src[s + 1];
    let e1 = src[s + 2];
    let c1 = src[s + 3];
    let prod = mul2(r, c0, c1);
    (e0 + prod.0, e1 + prod.1)
}

/// Two independent `r · x` products via the two-lane NEON PMULL kernel where
/// available.
#[inline(always)]
fn mul2(r: F128, x0: F128, x1: F128) -> (F128, F128) {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        // SAFETY: the `aes` target feature is guaranteed by the cfg gate.
        let p = unsafe { crate::field::gf2_128::aarch64::ghash_mul_vec2_neon([r, r], [x0, x1]) };
        (p[0], p[1])
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        (r * x0, r * x1)
    }
}

/// Final binding of a 2-entry projective table: `c0 + r · c1`.
pub(super) fn final_bind_projective(a: &[F128], r: F128) -> F128 {
    debug_assert_eq!(a.len(), 2);
    a[0] + r * a[1]
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
        fn vec(&mut self, n: usize) -> Vec<F128> {
            (0..n).map(|_| self.f128()).collect()
        }
    }

    #[test]
    fn twist_is_an_involution() {
        let mut rng = Rng::new(1);
        let orig = rng.vec(16);
        let mut t = orig.clone();
        twist_lowest_in_place(&mut t);
        assert_ne!(t, orig, "twist should change the table");
        twist_lowest_in_place(&mut t);
        assert_eq!(t, orig, "twisting twice must be the identity in char 2");
    }

    #[test]
    fn production_parallel_round_matches_boolean_round() {
        for log_n in 10..=12 {
            let n = 1usize << log_n;
            let mut rng = Rng::new(0xB16B_00B5 + log_n as u64);
            let a = rng.vec(n);
            let b = rng.vec(n);
            let r_fold = rng.f128();
            let mut r_next = vec![F128::ONE; log_n - 1];
            r_next[1..].copy_from_slice(&rng.vec(log_n - 2));

            let mut ba = vec![F128::ZERO; n / 2];
            let mut bb = vec![F128::ZERO; n / 2];
            let m_base = crate::zerocheck::multilinear::fold_and_compute_round_pair_into(
                &a, &b, &mut ba, &mut bb, r_fold, &r_next,
            );

            let mut ta = a.clone();
            let mut tb = b.clone();
            twist_lowest_in_place(&mut ta);
            twist_lowest_in_place(&mut tb);
            let mut pa = vec![F128::ZERO; n / 2];
            let mut pb = vec![F128::ZERO; n / 2];
            let m_projective =
                crate::zerocheck::multilinear::fold_and_compute_round_pair_projective_into(
                    &ta, &tb, &mut pa, &mut pb, r_fold, &r_next,
                );

            assert_eq!(m_base, m_projective, "message mismatch at log_n={log_n}");
            twist_lowest_in_place(&mut pa);
            twist_lowest_in_place(&mut pb);
            assert_eq!(ba, pa, "a output mismatch at log_n={log_n}");
            assert_eq!(bb, pb, "b output mismatch at log_n={log_n}");
        }
    }
}
