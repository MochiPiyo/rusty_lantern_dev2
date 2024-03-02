
use std::marker::PhantomData;
use crate::{autograd::Context, dtype::{Dtype, Shape}, logger::LOGGER, nten::{get_new_nten_id, Nten, NtenID}, tensor::Tensor2d};
use super::{get_new_fn_edge_id, FnEdge, FnEdgeID};


/*

FnEdgeは全デバイスで共通，FnEdgeの内部で分岐する

*/

// this FnEdge's front fn is implemented at Tensor2d
// fn add() @Tensor2d

// reference impl
pub fn add<const R: usize, const C: usize, T: Dtype>(lhs: &Nten, rhs: &Nten) -> Nten {
    let new_id: NtenID = get_new_nten_id(false);
    let add2d: Add2d<R, C, T> = Add2d::<R, C, T> {
        id: get_new_fn_edge_id(),
        name: format!("Add2d<{}, {}, {}>", R, C, T::type_name()),
        sources: vec![lhs.creator.clone(), rhs.creator.clone()],
        input1_id: lhs.id,
        input2_id: rhs.id,
        output_id: new_id,
        _marker: PhantomData,
    };
    Nten {
        id: new_id,
        name: format!("auto created by Add2d<{}, {}, {}>", R, C, T::type_name()),
        creator: Box::new(add2d),
        shape: Shape::D2(R, C),
        val: None,
        grad: None,
    }
}

#[derive(Clone)]
pub struct Add2d<const R: usize, const C: usize, T> {
    pub id: FnEdgeID,
    pub name: String,
    pub sources: Vec<Box<dyn FnEdge>>,

    pub input1_id: NtenID,
    pub input2_id: NtenID,
    pub output_id: NtenID,

    pub _marker: PhantomData<T>,
}
impl<const R: usize, const C: usize, T: Dtype> FnEdge for Add2d<R, C, T> {
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
        LOGGER.debug(format!("Add2d<{},{},{}> forward", R, C, T::type_name()));
        let input1: Tensor2d<R, C, T> = ctx.get_val(&self.input1_id).to_typed2d().unwrap();
        let input2: Tensor2d<R, C, T> = ctx.get_val(&self.input2_id).to_typed2d().unwrap();

        let output = input1.add(&input2);
        LOGGER.debug(format!("{:?}", &output));

        ctx.insert_val(&self.output_id, output.to_untyped());
    }

    fn backward(&self, ctx: &mut Context) {
        LOGGER.debug(format!("Add2d<{},{},{}> backward", R, C, T::type_name()));
        let dout = ctx.get_grad(&self.output_id);

        ctx.add_assign_grad(&self.input1_id, &dout);
        ctx.add_assign_grad(&self.input2_id, &dout);
    }
}