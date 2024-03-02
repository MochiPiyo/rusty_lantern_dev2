use std::{marker::PhantomData};

use crate::{autograd::VarStore, dtype::{Dtype, Shape}, fn_edge::{get_new_fn_edge_id, Add2d, FnEdge, HumanCreatedFnEdge}, tensor::Tensor2d};

use super::{get_new_nten_id, Nten, NtenID};

pub struct Nten2d<const R: usize, const C: usize, T=f32> {
    pub id: NtenID,
    pub name: String,
    pub creator: Box<dyn FnEdge>,

    pub(crate) val: Option<Tensor2d<R, C, T>>,
    pub(crate) grad: Option<Tensor2d<R, C, T>>,

    // dummy T
    // PhantomData will not include in struct instance after compile
    pub _marker: PhantomData<T>,
}
impl<const R: usize, const C: usize, T: Dtype> Nten2d<R, C, T> {
    pub fn new_from_val(val: Tensor2d<R, C, T>) -> Self {
        Self {
            id: get_new_nten_id(false),
            name: "no_name".to_string(),
            creator: Box::new(HumanCreatedFnEdge::new()),
            val: Some(val),
            grad: None,
            _marker: PhantomData,
        }
    }

    pub fn to_untyped(self) -> Nten {
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
        Nten {
            id: self.id,
            name: self.name,
            creator: self.creator,
            shape: Shape::D2(R, C),
            val: val_untyped,
            grad: grad_untyped,
        }
    }

    
    pub fn as_parameter(mut self, vs: &mut VarStore) -> Self {
        let mut val_untyped = if let Some(val) = self.val {
            Some(val.to_untyped())
        } else {
            None
        };
        let mut grad_untyped = if let Some(grad) = self.grad {
            Some(grad.to_untyped())
        } else {
            None
        };

        let clone_move_data = Nten {
            id: self.id.clone(),
            name: self.name.clone(),
            creator: self.creator.clone(),
            shape: Shape::D2(R, C),
            // Option::takeは所有権を移動させて元のOptionをNoneにする
            val: val_untyped.take(),
            grad: grad_untyped.take(),
        };

        vs.resister(clone_move_data);

        Nten2d::<R, C, T> {
            id: self.id,
            name: self.name,
            creator: self.creator,
            val: None,
            grad: None,
            _marker: PhantomData,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        let new_id: NtenID = get_new_nten_id(false);
        let add2d: Add2d<R, C, T> = Add2d::<R, C, T> {
            id: get_new_fn_edge_id(),
            name: format!("auto created by Add2d<{}, {}, {}>", R, C, T::type_name()),
            sources: vec![self.creator.clone(), other.creator.clone()],
            input1_id: self.id,
            input2_id: other.id,
            output_id: new_id,
            _marker: PhantomData,
        };
        Self {
            id: new_id,
            name: "add2d".to_string(),
            creator: Box::new(add2d),
            val: None,
            grad: None,
            _marker: PhantomData,
        }
    }
}