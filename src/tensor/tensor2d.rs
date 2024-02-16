use crate::{dtype::{Dtype, Shape}, fn_edge::{get_new_fn_edge_id, Add2d, FnEdge, HumanCreatedFnEdge}, raw_tensor::RawTensor2d};

use super::{get_new_tensor_id, Tensor, TensorID};

pub struct Tensor2d<const R: usize, const C: usize, T> {
    pub id: TensorID,
    pub name: String,
    pub creator: Box<dyn FnEdge>,

    pub val: Option<RawTensor2d<R, C, T>>,
    pub grad: Option<RawTensor2d<R, C, T>>,
    pub dummy: T
}
impl<const R: usize, const C: usize, T: Dtype> Tensor2d<R, C, T> {
    pub fn new_from_val(val: RawTensor2d<R, C, T>) -> Self {
        Self {
            id: get_new_tensor_id(false),
            name: "no_name".to_string(),
            creator: Box::new(HumanCreatedFnEdge::new()),
            val: Some(val),
            grad: None,
            dummy: T::default(),
        }
    }

    pub fn to_untyped(self) -> Tensor {
        let val_untyped = if let Some(val) = self.val {
            Some(val.to_untyped())
        } else {
            None
        };
        let grad_untyped = if let Some(grad) = self.grad {
            Some(grad.to_untyped())
        } else {
            None
        };
        Tensor {
            id: self.id,
            name: self.name,
            creator: self.creator,
            shape: Shape::D2(R, C),
            val: val_untyped,
            grad: grad_untyped,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        let new_id: TensorID = get_new_tensor_id(false);
        let add2d: Add2d<R, C, T> = Add2d::<R, C, T> {
            id: get_new_fn_edge_id(),
            sources: vec![self.creator.clone(), other.creator.clone()],
            input1_id: self.id,
            input2_id: other.id,
            output_id: new_id,
            dummy: T::default()
        };
        Self {
            id: new_id,
            name: "add2d".to_string(),
            creator: Box::new(add2d),
            val: None,
            grad: None,
            dummy: T::default()
        }
    }
}