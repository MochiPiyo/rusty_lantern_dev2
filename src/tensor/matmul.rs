use std::{marker::PhantomData, sync::{Arc, RwLock}};

use crate::dtype::{Dtype, Shape};

use super::{Storage, Tensor2d};

pub fn matmul<const N: usize, const M: usize, const O: usize, T: Dtype>(
    lhs: &Tensor2d<N, M, T>,
    rhs: &Tensor2d<M, O, T>,
) -> Tensor2d<N, O, T> {
    let lhs_storage = lhs.storage();
    let rhs_storage = rhs.storage();
    let result_storage =
        Storage::matmul(&lhs_storage, Shape::D2(N, M), &rhs_storage, Shape::D2(M, O));
    Tensor2d::<N, O, T> {
        name: format!("{} x {}", lhs.name, rhs.name),
        storage: Arc::new(RwLock::new(result_storage)),
        _marker: PhantomData,
    }
}
