use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

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
/// ```
macro_rules! fe {
    ($value:expr) => {
        $crate::field_element::FieldElement::from($value)
    };
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
