/// Helper utilities for generating deterministic byte patterns used in examples
/// and tests.
#[must_use]
pub fn patterned_bytes<const LEN: usize>(
    multiplier: u8,
    addend: u8,
) -> [u8; LEN] {
    let mut bytes = [0u8; LEN];
    fill_pattern(&mut bytes, multiplier, addend);
    bytes
}

/// Fill the provided buffer with the pattern `(index * multiplier + addend) mod
/// 256`.
#[allow(clippy::cast_possible_truncation)] // intentional wrapping for patterned fixtures
pub fn fill_pattern(bytes: &mut [u8], multiplier: u8, addend: u8) {
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = (i as u8).wrapping_mul(multiplier).wrapping_add(addend);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterned_bytes_produces_expected_sequence() {
        let bytes = patterned_bytes::<8>(3, 5);
        assert_eq!(bytes, [5, 8, 11, 14, 17, 20, 23, 26]);
    }

    #[test]
    fn fill_pattern_wraps_on_overflow() {
        let mut buf = [0u8; 4];
        fill_pattern(&mut buf, 200, 200);
        assert_eq!(buf, [200, 144, 88, 32]);
    }

    #[test]
    fn patterned_bytes_respects_length() {
        let bytes = patterned_bytes::<0>(1, 2);
        assert!(bytes.is_empty());
    }
}
