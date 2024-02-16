use std::collections::{HashMap, HashSet};

use crate::{fn_edge::{FnEdge, FnEdgeID}, raw_tensor::RawTensor, tensor::{Tensor, TensorID, TensorTrait}};


pub struct Context {
    store: HashMap<TensorID, Tensor>,
}
impl Context {
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }
    pub fn get_val(&mut self, id: &TensorID) -> Result<RawTensor, String> {
        if let Some(tensor) = self.store.remove(id) {
            if let Some(val) = tensor.val {
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

    pub fn get_grad(&mut self, id: &TensorID) -> Result<RawTensor, String> {
        if let Some(tensor) = self.store.remove(id) {
            if let Some(grad) = tensor.grad {
                Ok(grad)
            } else {
                Err(format!("Tensor id: {} has no val", id))
            }
        } else {
            Err(format!("Tensor id: {} not found in Context", id))
        }
    }

    pub fn add_assign_grad(&mut self, id: &TensorID, raw_tensor: &RawTensor) {
        if let Some(tensor) = self.store.get_mut(id) {
            if let Some(old_grad) = &tensor.grad {
                tensor.grad = Some(old_grad.add(&raw_tensor).unwrap());
            } else {
                tensor.grad = Some(raw_tensor.clone());
            }
        } else {
            panic!("Tensor id: {} not found in Context", id)
        }
    }
}

pub struct Autograd {
    already_executed: HashSet<FnEdgeID>,
    next_execute_index: usize,
    tape: Vec<Box<dyn FnEdge>>,

    ctx: Context,
}
impl Autograd {
    pub fn new() -> Self {
        Self {
            already_executed: HashSet::new(),
            next_execute_index: 0,
            tape: Vec::new(),

            ctx: Context::new(),
        }
    }
    // return value
    pub fn step_forward<const N: usize>(&mut self, results: [Tensor; N]) -> [Tensor; N] {

        // グラフ探索してテープを構築，self.tapeに追加される
        self._build_tape(&results);
        // 実行
        for i in self.next_execute_index..self.tape.len() {
            self.tape[i].forward(&mut self.ctx);
        }
        // 次回のために開始位置をずらす
        self.next_execute_index = self.tape.len();

        results
    }
    pub fn backward<'a>(&'a mut self, result: Tensor) -> &'a Context {
        if let None = result.grad {
            panic!("cannot start backward. grad of result tensor is None");
        }
        for i in (0..self.tape.len()).rev() {
            self.tape[i].backward(&mut self.ctx);
        }
        &self.ctx
    }

    pub fn _build_tape<const N: usize>(&mut self, results: &[Tensor; N]) {
        let mut already_seen = HashSet::new();
        let mut stack = Vec::new();
        for tensor in results.iter() {
            stack.push(tensor.creator.clone_box());
        }
        
        while let Some(node) = stack.pop() {
            // if it was 
            if self.already_executed.contains(&node.id()) {
                continue;
            }
            if !already_seen.contains(&node.id()) {
                already_seen.insert(node.id());
                // tape
                self.tape.push(node.clone_box());
                for input in node.inputs() {
                    stack.push(input.clone_box());
                }
            }
        }
    
        self.tape.reverse();
    }

}