
use std::marker::PhantomData;
use crate::{autograd::Context, dtype::{Dtype, Shape}, logger::LOGGER, nten::{get_new_nten_id, Nten, Nten2d, NtenID}, tensor::{self, Tensor2d}};
use super::{get_new_fn_edge_id, FnEdge, FnEdgeID};



// fnはnten matmulにある


#[derive(Clone)]
pub struct Matmul<const N: usize, const M: usize, const O: usize, T> {
    pub id: FnEdgeID,
    pub name: String,
    pub sources: Vec<Box<dyn FnEdge>>,

    pub lhs_id: NtenID,
    pub rhs_id: NtenID,
    pub output_id: NtenID,

    pub _marker: PhantomData<T>,
}
impl<const N: usize, const M: usize, const O: usize, T: Dtype> FnEdge for Matmul<N, M, O, T> {
    fn get_id(&self) -> FnEdgeID {
        self.id
    }
    fn sources(&self) -> Vec<Box<dyn FnEdge>> {
        self.sources.clone()
    }
    fn inputs(&self) -> Vec<NtenID> {
        vec![self.lhs_id, self.rhs_id]
    }
    fn clone_box(&self) -> Box<dyn FnEdge> {
        Box::new(self.clone())
    }


    fn forward(&self, ctx: &mut Context) {
        let lhs: Tensor2d<N, M, T> = ctx.get_val(&self.lhs_id).to_typed2d().unwrap();
        let rhs: Tensor2d<M, O, T> = ctx.get_val(&self.rhs_id).to_typed2d().unwrap();

        let out = tensor::matmul(&lhs, &rhs);

        ctx.insert_val(&self.output_id, out.to_untyped());
    }

    fn backward(&self, ctx: &mut Context) {
        let lhs: Tensor2d<N, M, T> = ctx.get_val(&self.lhs_id).to_typed2d().unwrap();
        let rhs: Tensor2d<M, O, T> = ctx.get_val(&self.rhs_id).to_typed2d().unwrap();

        let din: Tensor2d<N, O, T> = ctx.get_grad(&self.output_id).to_typed2d().unwrap();

        let dlhs = tensor::matmul(&din, &rhs.transpose());
        let drhs = tensor::matmul(&lhs.transpose(), &din);

        ctx.add_assign_grad(&self.lhs_id, &dlhs.to_untyped());
        ctx.add_assign_grad(&self.rhs_id, &drhs.to_untyped());
    }
}