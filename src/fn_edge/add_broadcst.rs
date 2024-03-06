
use std::marker::PhantomData;
use crate::{autograd::Context, dtype::{Dtype, Shape}, logger::LOGGER, nten::{get_new_nten_id, Nten, NtenID}, tensor::Tensor2d};
use super::{get_new_fn_edge_id, FnEdge, FnEdgeID};


/*

FnEdgeは全デバイスで共通，FnEdgeの内部で分岐する

*/

// this FnEdge's front fn is implemented at Tensor2d
// fn add_broadcast() @Nten2d

#[derive(Clone)]
pub struct AddBroadcast2d<const R: usize, const C: usize, T> {
    pub id: FnEdgeID,
    pub name: String,
    pub sources: Vec<Box<dyn FnEdge>>,

    pub input1_id: NtenID,
    pub input2_id: NtenID,
    pub output_id: NtenID,

    pub _marker: PhantomData<T>,
}
impl<const R: usize, const C: usize, T: Dtype> FnEdge for AddBroadcast2d<R, C, T> {
    fn get_id(&self) -> FnEdgeID {
        self.id
    }
    fn sources(&self) -> Vec<Box<dyn FnEdge>> {
        self.sources.clone()
    }
    fn inputs(&self) -> Vec<NtenID> {
        vec![self.input1_id, self.input2_id]
    }
    fn clone_box(&self) -> Box<dyn FnEdge> {
        Box::new(self.clone())
    }

    fn forward(&self, ctx: &mut Context) {
        LOGGER.debug(format!("AddBroadcast2d<{},{},{}> forward", R, C, T::type_name()));
        let input1: Tensor2d<R, C, T> = ctx.get_val_as_2d(&self.input1_id);
        let input2: Tensor2d<1, C, T> = ctx.get_val_as_2d(&self.input2_id);

        let output = input1.add_broadcast(&input2);

        ctx.insert_val(&self.output_id, output.to_untyped());
    }

    fn backward(&self, ctx: &mut Context) {
        LOGGER.debug(format!("AddBroadcast2d<{},{},{}> backward", R, C, T::type_name()));
        let dout = ctx.get_grad(&self.output_id);

        todo!()
    }
}