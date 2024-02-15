use crate::{fn_edge::FnEdge, raw_tensor::RawTensor, tensor::{Tensor, TensorID, TensorTrait}};


pub struct Context {
    store: std::collections::HashMap<TensorID, Tensor>,
}
impl Context {
    pub fn get_val<'a>(&'a self, id: &TensorID) -> Result<&'a RawTensor, String> {
        if let Some(tensor) = self.store.get(id) {
            if let Some(val) = &tensor.val {
                Ok(val)
            } else {
                Err(format!("Tensor id: {} has no val", id))
            }
        } else {
            Err(format!("Tensor id: {} not found in Context", id))
        }
    }
    pub fn insert_val(&mut self, id: &TensorID, raw_tensor: RawTensor)  {
        if let Some(tensor) = self.store.get_mut(id) {
            tensor.val = Some(raw_tensor);
        } else {
            panic!("Tensor id: {} not found in Context", id)
        }
    }
}

pub struct Autograd {

}
impl Autograd {
    pub fn new() -> Self {
        Self {

        }
    }
    // return value
    pub fn step_forward<const N: usize>(&mut self, result: [Tensor; N]) -> [Tensor; N] {
        todo!()
    }
    pub fn backward(&mut self, result: Tensor) -> Context {
        todo!()
    }

    fn _build_tape<T: TensorTrait>(last_node: T) {

    }

    fn _execute_forward(tape: &Vec<Box<dyn FnEdge>>, ctx: Context) {

    }
    fn _backward(tape: &Vec<Box<dyn FnEdge>>, ctx: Context) {

    }
}