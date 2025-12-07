use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[cfg(test)]
use proptest::{
    arbitrary::Arbitrary,
    strategy::{BoxedStrategy, Strategy},
};

/// Lightweight field element wrapper for `F_257`.
///
/// Provides modular addition/multiplication helpers without branching on feature
/// flags at call sites. Montgomery reduction is provided for consistency and
/// potential future speedups.
#[must_use]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct FieldElement(pub u16);

#[macro_export]
/// Construct a `FieldElement` modulo 257.
///
/// # Examples
/// ```
/// use swifft::fe;
///
/// let small = fe!(5);
/// let reduced = fe!(300); // 300 ≡ 43 (mod 257)
///
/// assert_eq!(small.value(), 5);
/// assert_eq!(reduced.value(), 43);
/// assert_eq!(fe!(1) + fe!(2), fe!(3));
/// assert_eq!(fe!(10) + fe!(20), fe!(30));
/// // Addition wraps modulo 257.
/// assert_eq!(fe!(200) + fe!(100), fe!(43));
/// ```
macro_rules! fe {
    ($value:expr) => {
        $crate::field_element::FieldElement::from($value)
    };
}

#[cfg(test)]
impl Arbitrary for FieldElement {
    type Parameters = ();
    type Strategy = BoxedStrategy<FieldElement>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        (0u16..=FieldElement::MAX)
            .prop_map(FieldElement::from)
            .boxed()
    }
}

/// Create a `FieldElement` from a numeric literal or expression.
///
/// ```
/// use swifft::fe;
///
/// let a = fe!(42);
/// let b = fe!(1 + 1);
/// assert_eq!(a.value(), 42);
/// assert_eq!(b.value(), 2);
/// ```
#[doc(inline)]
pub use crate::fe;

impl FieldElement {
    /// Prime modulus p.
    pub const P: u16 = 257;

    pub const BYTES: usize = 2;
    pub const MAX: u16 = Self::P - 1;
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);
    /// R = 2^16 for Montgomery reduction.
    const R_BITS: u32 = 16;
    const R_MASK: u32 = (1u32 << Self::R_BITS) - 1;
    /// -P^{-1} mod R for P = 257, R = 2^16.
    const N_PRIME: u16 = 255;

    #[inline]
    pub const fn new(v: u16) -> Self {
        Self(v % Self::P)
    }

    #[inline]
    #[must_use]
    pub const fn value(self) -> u16 {
        self.0
    }

    #[inline]
    #[must_use]
    pub const fn is_canonical(v: u16) -> bool {
        v < Self::P
    }

    #[inline]
    #[must_use]
    pub const fn from_canonical(v: u16) -> Option<Self> {
        if Self::is_canonical(v) {
            Some(Self(v))
        } else {
            None
        }
    }

    #[allow(dead_code)] // handy for future algebraic ops
    pub fn pow(self, mut exp: u16) -> Self {
        let mut base = self;
        let mut acc = FieldElement::ONE;

        while exp > 0 {
            if exp & 1 == 1 {
                acc *= base;
            }
            base *= base;
            exp >>= 1;
        }
        acc
    }

    #[allow(dead_code)] // handy for future algebraic ops
    pub fn inv(self) -> Self {
        // Fermat's little theorem: a^(p-2) mod p for prime p.
        self.pow(Self::P - 2)
    }

    /// Montgomery reduction for 16-bit modulus P = 257 with R = 2^16.
    #[inline]
    #[must_use]
    pub const fn montyred(x: u32) -> u16 {
        // m = (x * n') mod R
        let m = (x.wrapping_mul(Self::N_PRIME as u32)) & Self::R_MASK;

        // t = (x + m * P) / R
        let t = (x.wrapping_add(m * (Self::P as u32))) >> Self::R_BITS;
        let t16 = t as u16;

        if t16 >= Self::P {
            t16 - Self::P
        } else {
            t16
        }
    }
}

impl From<u16> for FieldElement {
    #[inline]
    fn from(v: u16) -> Self {
        FieldElement::new(v)
    }
}

impl From<FieldElement> for u16 {
    #[inline]
    fn from(fe: FieldElement) -> Self {
        fe.0
    }
}

impl Add for FieldElement {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let sum = self.0 + rhs.0;
        if sum >= Self::P {
            FieldElement(sum - Self::P)
        } else {
            FieldElement(sum)
        }
    }
}

impl AddAssign for FieldElement {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Mul for FieldElement {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let prod = FieldElement::montyred(u32::from(self.0) * u32::from(rhs.0));
        FieldElement(prod)
    }
}

impl MulAssign for FieldElement {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Sub for FieldElement {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        if self.0 >= rhs.0 {
            FieldElement(self.0 - rhs.0)
        } else {
            FieldElement(self.0 + Self::P - rhs.0)
        }
    }
}

impl SubAssign for FieldElement {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for FieldElement {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        if self.0 == 0 {
            FieldElement::ZERO
        } else {
            FieldElement(Self::P - self.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn new_value_and_macro_reduction() {
        let from_new = FieldElement::new(300);
        assert_eq!(from_new.value(), 43);

        let from_macro = fe!(1 + 2);
        assert_eq!(from_macro.value(), 3);
        assert_eq!(fe!(300), fe!(43));
    }

    #[test]
    fn canonical_helpers() {
        assert!(FieldElement::is_canonical(0));
        assert!(FieldElement::is_canonical(FieldElement::MAX));
        assert!(!FieldElement::is_canonical(FieldElement::P));

        assert_eq!(FieldElement::from_canonical(42), Some(fe!(42)));
        assert_eq!(FieldElement::from_canonical(999), None);
    }

    #[test]
    fn conversions_and_constants() {
        let fe_from_u16 = FieldElement::from(300u16);
        assert_eq!(fe_from_u16, fe!(43));

        let as_u16: u16 = fe!(255).into();
        assert_eq!(as_u16, 255);

        assert_eq!(FieldElement::ZERO, fe!(0));
        assert_eq!(FieldElement::ONE, fe!(1));
        assert_eq!(FieldElement::BYTES, 2);
    }

    #[test]
    fn pow_and_inv_cover_edge_cases() {
        assert_eq!(fe!(7).pow(0), FieldElement::ONE);
        assert_eq!(fe!(7).pow(1), fe!(7));
        assert_eq!(fe!(2).pow(8), fe!(256)); // 2^8 mod 257

        let inv = fe!(3).inv();
        assert_eq!(inv, fe!(86)); // 3 * 86 = 258 ≡ 1 (mod 257)
        assert_eq!(fe!(3) * inv, FieldElement::ONE);
    }

    #[test]
    fn montyred_reduces_modulus() {
        assert_eq!(FieldElement::montyred(0), 0);
        assert_eq!(FieldElement::montyred(FieldElement::P as u32), 0);
        assert_eq!(FieldElement::montyred(512), 255);
    }

    #[test]
    fn add_and_add_assign_wrap() {
        let mut a = fe!(200);
        assert_eq!(a + fe!(100), fe!(43));

        a += fe!(100);
        assert_eq!(a, fe!(43));
    }

    #[test]
    fn sub_and_sub_assign_wrap() {
        let mut a = fe!(10);
        assert_eq!(a - fe!(20), fe!(247));

        a -= fe!(20);
        assert_eq!(a, fe!(247));
    }

    #[test]
    fn mul_and_mul_assign_modular() {
        let mut a = fe!(13);
        let b = fe!(7);

        assert_eq!(a * b, fe!(91));
        a *= b;
        assert_eq!(a, fe!(91));
    }

    #[test]
    fn negation_handles_zero_and_non_zero() {
        assert_eq!(-FieldElement::ZERO, FieldElement::ZERO);
        assert_eq!(-fe!(5), fe!(252)); // 257 - 5
    }

    #[test]
    fn add_sub_mul_basic() {
        let a = fe!(200);
        let b = fe!(100);
        assert_eq!(a + b, fe!(43)); // 300 mod 257
        assert_eq!(a - b, fe!(100));
        assert_eq!(b - a, fe!(157)); // wrap
        assert_eq!(a * b, fe!(211));
    }

    proptest! {
        #[test]
        fn add_commutative(a: FieldElement, b: FieldElement) {
            prop_assert_eq!(a + b, b + a);
        }

        #[test]
        fn mul_distributes_over_add(a: FieldElement, b: FieldElement, c: FieldElement) {
            prop_assert_eq!(a * (b + c), (a * b) + (a * c));
        }

        #[test]
        fn pow_matches_naive(a in 0u16..=256, e in 0u16..=512) {
            let a_fe = fe!(a);
            let fast = a_fe.pow(e).value();
            let mut naive = 1u32;
            for _ in 0..e {
                naive = (naive * a as u32) % (FieldElement::P as u32);
            }
            prop_assert_eq!(fast, naive as u16);
        }

        #[test]
        fn montyred_matches_mod_reduce(x in 0u32..100_000u32) {
            let reduced = FieldElement::montyred(x);
            prop_assert!(reduced < FieldElement::P);
            prop_assert_eq!(reduced as u32, x % (FieldElement::P as u32));
        }

        #[test]
        fn neg_and_inv_properties(a in 1u16..=256) { // avoid zero for inverse
            let fe = fe!(a);
            prop_assert_eq!(fe + (-fe), fe!(0));
            let inv = fe.inv();
            prop_assert_eq!(fe * inv, fe!(1));
        }
    }
}
