
use crate::{dtype::Dtype, raw_tensor::{RawTensor, RawTensor2d}, tensor::TensorID, autograd::Context};

use super::FnEdge;


/*

FnEdgeは全デバイスで共通，FnEdgeの内部で分岐する

*/

// this FnEdge's front fn is implemented at Tensor2d
// fn add() @Tensor2d


#[derive(Clone)]
pub struct Add2d<const R: usize, const C: usize, T> {
    pub input1_id: TensorID,
    pub input2_id: TensorID,
    pub output_id: TensorID,
    pub dummy: T,
}
impl<const R: usize, const C: usize, T: Dtype> FnEdge for Add2d<R, C, T> {
    fn forward(&self, ctx: &mut Context) {
        let input1_non_typed: &RawTensor = ctx.get_val(&self.input1_id).unwrap();
        let input1: &RawTensor2d<R, C, T> = input1_non_typed.to_typed2d().unwrap();
        let input2: &RawTensor2d<R, C, T> = ctx.get_val(&self.input2_id).unwrap().to_typed2d().unwrap();

        let output = input1.add(input2);

        ctx.insert_val(&self.output_id, output.to_untyped());
    }
    fn backward(&self, ctx: &mut Context) {

    }
}