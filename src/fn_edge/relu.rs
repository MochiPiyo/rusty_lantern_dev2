use std::marker::PhantomData;
use crate::{autograd::Context, dtype::{Dtype, Shape}, logger::LOGGER, nten::{get_new_nten_id, Nten, Nten2d, NtenID}, tensor::Tensor2d};
use super::{get_new_fn_edge_id, FnEdge, FnEdgeID};




#[derive(Clone)]
pub struct Relu2d<const R: usize, const C: usize, T> {
    pub id: FnEdgeID,
    pub name: String,
    pub sources: Vec<Box<dyn FnEdge>>,

    pub input_id: NtenID,
    pub output_id: NtenID,
    pub mask_cach_id: NtenID,

    pub _marker: PhantomData<T>
}
impl<const R: usize, const C: usize, T: Dtype> FnEdge for Relu2d<R, C, T> {
    fn get_id(&self) -> super::FnEdgeID {
        self.id
    }

    fn name(&self) -> String {
        format!("Relu<{}, {}, {}>", R, C, T::type_name())
    }

    fn sources(&self) -> Vec<Box<dyn FnEdge>> {
        self.sources.clone()
    }

    fn clone_box(&self) -> Box<dyn FnEdge> {
        Box::new(self.clone())
    }

    fn forward(&self, ctx: &mut crate::autograd::Context) {
        let input: Tensor2d<R, C, T> = ctx.get_val_as_2d(&self.input_id);

        let mask: Tensor2d<R, C, bool> = input.select_smaller_than(T::from_f32(0.0));
        ctx.insert_tensor(&self.mask_cach_id, mask.to_untyped());

        let output: Tensor2d<R, C, T> = input.replace_scalar_where(&mask, T::from_f32(0.0));

        ctx.insert_val(&self.output_id, output.to_untyped());
    }

    fn backward(&self, ctx: &mut crate::autograd::Context) {
        let din: Tensor2d<R, C, T> = ctx.get_val_as_2d(&self.output_id);

        let mask: Tensor2d<R, C, bool> = ctx.get_tensor_as_2d(&self.mask_cach_id);
        let dout: Tensor2d<R, C, T> = din.replace_scalar_where(&mask, T::from_f32(0.0));

        ctx.add_assign_grad(&self.output_id, &dout.to_untyped());
    }
}