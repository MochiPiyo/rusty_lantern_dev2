use std::marker::PhantomData;

use crate::{dtype::Dtype, fn_edge::{get_new_fn_edge_id, Matmul}};

use super::{get_new_nten_id, Nten2d};

pub fn matmul<const N: usize, const M: usize, const O: usize, T: Dtype>
    (lhs: &Nten2d<N, M, T>, rhs: &Nten2d<M, O, T>) -> Nten2d<N, O, T> {
    let new_id = get_new_nten_id();
    let matmul = Matmul::<N, M, O, T> {
        id: get_new_fn_edge_id(),
        name: format!("Matmul<{}, {}, {}, {}>", N, M, O, T::type_name()),
        sources: vec![lhs.creator.clone(), rhs.creator.clone()],
        lhs_id: lhs.id,
        rhs_id: rhs.id,
        output_id: new_id,
        _marker: PhantomData,
    };
    Nten2d {
        id: new_id,
        name: format!("auto created by Matmul<{}, {}, {}, {}>", N, M, O, T::type_name()),
        creator: Box::new(matmul),
        val: None,
        grad: None,
        _marker: PhantomData,
    }
}