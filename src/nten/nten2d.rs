use std::marker::PhantomData;

use crate::{autograd::VarStore, dtype::{Dtype, Shape}, fn_edge::{get_new_fn_edge_id, Add2d, AddBroadcast2d, FnEdge, HumanCreatedFnEdge}, logger::LOGGER, tensor::Tensor2d};

use super::{get_new_nten_id, relu::Relu2d, Nten, NtenID};

#[derive(Clone)]
pub struct Nten2d<const R: usize, const C: usize, T> {
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
            id: get_new_nten_id(),
            name: "no_name".to_string(),
            creator: Box::new(HumanCreatedFnEdge::new()),
            val: Some(val),
            grad: None,
            _marker: PhantomData,
        }
    }

    pub fn name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
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

    pub fn type_name(&self) -> String {
        format!("Nten2d<{}, {}, {}>", R, C, T::type_name())
    }

    
    pub fn as_parameter(self, vs: &mut VarStore) -> Self {
        if let None = self.val {
            LOGGER.error(format!("{}::as_parameter() >> nten id: {}, name: '{}' self.val is None. \
            parameter val must have Some.", self.type_name(), self.id, self.name));
            panic!();
        }
        if let Some(_) = self.grad {
            LOGGER.warning(format!("{}::as_parameter() >> nten id: {}, name: '{}' expected grad is None but has some. \
                you may forgot clear grad or reuse nten in iteration.", self.type_name(), self.id, self.name));
        }

        let to_resistor = self.clone();
        vs.resister_parameter(to_resistor.to_untyped());

        self
    }

    pub fn as_input(self, vs: &mut VarStore) -> Self {
        if let None = self.val {
            LOGGER.error(format!("{}::as_input() >> nten id: {}, name: '{}' self.val is None. \
            parameter val must have Some.", self.type_name(), self.id, self.name));
            panic!();
        }
        if let Some(_) = self.grad {
            LOGGER.warning(format!("{}::as_input() >> nten id: {}, name: '{}' expected grad is None but has some. \
                you may forgot clear grad or reuse nten in iteration.", self.type_name(), self.id, self.name));
        }

        let to_resistor = self.clone();
        vs.resister_input(to_resistor.to_untyped());

        self
    }

    pub fn add(&self, other: &Self) -> Self {
        let new_id: NtenID = get_new_nten_id();
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

    pub fn add_broadcast(&self, bias: &Nten2d<1, C, T>) -> Self {
        let new_id = get_new_nten_id();
        let fn_edge = AddBroadcast2d::<R, C, T> {
            id: get_new_fn_edge_id(),
            name: format!("auto created by Add2d<{}, {}, {}>", R, C, T::type_name()),
            sources: vec![self.creator.clone(), bias.creator.clone()],
            weight_id: self.id,
            bias_id: bias.id,
            output_id: new_id,
            _marker: PhantomData,
        };
        Self {
            id: new_id,
            name: "add broadcast".to_string(),
            creator: Box::new(fn_edge),
            val: None,
            grad: None,
            _marker: PhantomData,
        }
    }

    
    pub fn relu(&self) -> Nten2d<R, C, T> {
        let new_id = get_new_nten_id();
        let relu = Relu2d::<R, C, T> {
            id: get_new_fn_edge_id(),
            name: format!("Relu2d<{}, {}, {}>", R, C, T::type_name()),
            sources: vec![self.creator.clone()],
            input_id: self.id,
            output_id: new_id,
            mask_cach_id: get_new_nten_id(),
            _marker: PhantomData,
        };
        Nten2d {
            id: new_id,
            name: format!("auto created by Relu<{}, {}, {}>", R, C, T::type_name()),
            creator: Box::new(relu),
            val: None,
            grad: None,
            _marker: PhantomData,
        }
    }
}