use crate::{
    fe,
    field_element::FieldElement,
    math::{M, N},
    state::{Key, Message},
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub(crate) trait CompressionBackend {
    fn transform_columns(msg: &Message) -> [[FieldElement; N]; M];
    fn reduce_columns(
        key: &Key,
        columns: &[[FieldElement; N]; M],
    ) -> [FieldElement; N];
}

#[cfg(feature = "parallel")]
pub(crate) struct ParallelBackend;

#[cfg(feature = "parallel")]
impl CompressionBackend for ParallelBackend {
    fn transform_columns(msg: &Message) -> [[FieldElement; N]; M] {
        let mut cols = [[FieldElement::ZERO; N]; M];
        cols.par_iter_mut()
            .enumerate()
            .for_each(|(j, slot)| *slot = msg.transform_column(j));
        cols
    }

    fn reduce_columns(
        key: &Key,
        columns: &[[FieldElement; N]; M],
    ) -> [FieldElement; N] {
        let mut out = [FieldElement::ZERO; N];
        out.par_iter_mut().enumerate().for_each(|(i, z_i)| {
            let acc = columns.iter().enumerate().fold(
                FieldElement::ZERO,
                |acc, (j, y_col)| {
                    let a_ij = fe!(u16::from(key.0[j * N + i]));
                    acc + a_ij * y_col[i]
                },
            );
            *z_i = acc;
        });
        out
    }
}

#[cfg(not(feature = "parallel"))]
pub(crate) struct ScalarBackend;

#[cfg(not(feature = "parallel"))]
impl CompressionBackend for ScalarBackend {
    fn transform_columns(msg: &Message) -> [[FieldElement; N]; M] {
        core::array::from_fn(|j| msg.transform_column(j))
    }

    fn reduce_columns(
        key: &Key,
        columns: &[[FieldElement; N]; M],
    ) -> [FieldElement; N] {
        core::array::from_fn(|i| {
            columns.iter().enumerate().fold(
                FieldElement::ZERO,
                |acc, (j, y_col)| {
                    let a_ij = fe!(u16::from(key.0[j * N + i]));
                    acc + a_ij * y_col[i]
                },
            )
        })
    }
}

#[cfg(feature = "parallel")]
pub(crate) type Backend = ParallelBackend;

#[cfg(not(feature = "parallel"))]
pub(crate) type Backend = ScalarBackend;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        math, pattern::patterned_bytes, Block, Key, State, BLOCK_LEN, KEY_LEN,
        STATE_LEN,
    };

    fn reduce_columns_reference(
        key: &Key,
        columns: &[[FieldElement; N]; M],
    ) -> [FieldElement; N] {
        core::array::from_fn(|i| {
            columns.iter().enumerate().fold(
                FieldElement::ZERO,
                |acc, (j, y_col)| {
                    let a_ij = fe!(u16::from(key.0[j * N + i]));
                    acc + a_ij * y_col[i]
                },
            )
        })
    }

    #[test]
    fn backend_reduce_matches_reference() {
        let key = Key(patterned_bytes::<KEY_LEN>(9, 4));
        let state = State(patterned_bytes::<STATE_LEN>(2, 1));
        let block = Block(patterned_bytes::<BLOCK_LEN>(7, 3));

        let msg = Message::new(&state, &block);
        let columns = Backend::transform_columns(&msg);

        let expected = reduce_columns_reference(&key, &columns);
        let actual = Backend::reduce_columns(&key, &columns);
        assert_eq!(actual, expected);
    }

    #[cfg(not(feature = "parallel"))]
    #[test]
    fn scalar_backend_transform_and_reduce_match_reference() {
        let key = Key(patterned_bytes::<KEY_LEN>(4, 9));
        let state = State(patterned_bytes::<STATE_LEN>(6, 2));
        let block = Block(patterned_bytes::<BLOCK_LEN>(3, 7));

        let msg = Message::new(&state, &block);
        let columns = ScalarBackend::transform_columns(&msg);

        for (j, col) in columns.iter().enumerate() {
            assert_eq!(*col, msg.transform_column(j));
        }

        let reduced = ScalarBackend::reduce_columns(&key, &columns);
        let expected = reduce_columns_reference(&key, &columns);
        assert_eq!(reduced, expected);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_backend_transform_and_reduce_match_reference() {
        let key = Key(patterned_bytes::<KEY_LEN>(4, 9));
        let state = State(patterned_bytes::<STATE_LEN>(6, 2));
        let block = Block(patterned_bytes::<BLOCK_LEN>(3, 7));

        let msg = Message::new(&state, &block);
        let columns = ParallelBackend::transform_columns(&msg);

        for (j, col) in columns.iter().enumerate() {
            assert_eq!(*col, msg.transform_column(j));
        }

        let reduced = ParallelBackend::reduce_columns(&key, &columns);
        let expected = reduce_columns_reference(&key, &columns);
        assert_eq!(reduced, expected);
    }

    #[test]
    fn transform_matches_direct_math() {
        let state_bytes = patterned_bytes::<STATE_LEN>(1, 0);
        let block_bytes = patterned_bytes::<BLOCK_LEN>(1, 5);
        let msg =
            Message::new(&State::from(state_bytes), &Block::from(block_bytes));
        let columns = Backend::transform_columns(&msg);

        for (j, col) in columns.iter().enumerate() {
            let expected = math::transform(&msg.extract_column_bits(j));
            assert_eq!(col.map(FieldElement::value), expected);
        }
    }
}
