
use std::{fmt::format, marker::PhantomData};
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
    fn name(&self) -> String {
        format!("Matmul<{}, {}, {}, {}> that means matmul of Nten<{}, {}, {}> and Nten<{}, {}, {}>",
            N, M, O, T::type_name(),
            N, M, T::type_name(),
            M, O, T::type_name()
        )
    }
    fn sources(&self) -> Vec<Box<dyn FnEdge>> {
        self.sources.clone()
    }
    fn clone_box(&self) -> Box<dyn FnEdge> {
        Box::new(self.clone())
    }


    fn forward(&self, ctx: &mut Context) {
        ctx.varstore.print_all_contents_id();
        let lhs: Tensor2d<N, M, T> = ctx.get_val_as_2d(&self.lhs_id);
        let rhs: Tensor2d<M, O, T> = ctx.get_val_as_2d(&self.rhs_id);

        let out = tensor::matmul(&lhs, &rhs);

        ctx.insert_val(&self.output_id, out.to_untyped());
    }

    fn backward(&self, ctx: &mut Context) {
        let lhs: Tensor2d<N, M, T> = ctx.get_val_as_2d(&self.lhs_id);
        let rhs: Tensor2d<M, O, T> = ctx.get_val_as_2d(&self.rhs_id);

        let din: Tensor2d<N, O, T> = ctx.get_grad_as_2d(&self.output_id);

        let dlhs = tensor::matmul(&din, &rhs.transpose());
        let drhs = tensor::matmul(&lhs.transpose(), &din);

        ctx.add_assign_grad(&self.lhs_id, &dlhs.to_untyped());
        ctx.add_assign_grad(&self.rhs_id, &drhs.to_untyped());
    }
}