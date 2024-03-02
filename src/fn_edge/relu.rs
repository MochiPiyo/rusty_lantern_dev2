use std::marker::PhantomData;
use crate::{autograd::Context, dtype::{Dtype, Shape}, logger::LOGGER, nten::{get_new_nten_id, Nten, Nten2d, NtenID}, tensor::Tensor2d};
use super::{get_new_fn_edge_id, FnEdge, FnEdgeID};


// とりあえず2dを使う。
pub fn relu<const N: usize, T: Dtype>(nten: Nten2d<1, N>) -> Nten2d<1, N> {
    let new_id = get_new_nten_id(false);
    let relu = Relu::<N, T> {
        id: get_new_fn_edge_id(),
        name: format!("Relu<{}, {}>", N, T::type_name()),
        sources: vec![nten.creator],
        input_id: nten.id,
        output_id: new_id,
        _marker: PhantomData,
    };
    Nten2d {
        id: new_id,
        name: format!("auto created by Relu<{}, {}>", N, T::type_name()),
        creator: Box::new(relu),
        val: None,
        grad: None,
        _marker: PhantomData,
    }
}

#[derive(Clone)]
pub struct Relu<const N: usize, T> {
    pub id: FnEdgeID,
    pub name: String,
    pub sources: Vec<Box<dyn FnEdge>>,

    pub input_id: NtenID,
    pub output_id: NtenID,

    _marker: PhantomData<T>
}
impl<const N: usize, T: Dtype> FnEdge for Relu<N, T> {
    fn get_id(&self) -> super::FnEdgeID {
        self.id
    }

    fn sources(&self) -> Vec<Box<dyn FnEdge>> {
        self.sources.clone()
    }

    fn inputs(&self) -> Vec<NtenID> {
        vec![self.input_id]
    }

    fn clone_box(&self) -> Box<dyn FnEdge> {
        Box::new(self.clone())
    }

    fn forward(&self, ctx: &mut crate::autograd::Context) {
        todo!()
    }

    fn backward(&self, ctx: &mut crate::autograd::Context) {
        todo!()
    }
}