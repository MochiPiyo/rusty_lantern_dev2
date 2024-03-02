use std::{collections::{HashMap, HashSet}, fmt::format};

use crate::{
    fn_edge::{DummyFnEdge, FnEdge, FnEdgeID, HumanCreatedFnEdge}, logger::LOGGER, nten::{self, Nten, NtenID, NtenTrait}, tensor::Tensor
};

pub struct VarStore {
    pub(crate) body: HashMap<NtenID, Nten>,
}
impl VarStore {
    pub fn new() -> Self {
        Self {
            body: HashMap::new(),
        }
    }

    fn clone_nten(&self, id: &NtenID) -> Nten {
        self.body.get(id).unwrap().clone()
    }

    fn remove(&mut self, id: &NtenID) -> Option<Nten> {
        self.body.remove(id)
    }
    fn get_mut(&mut self, id: &NtenID) -> Option<&mut Nten> {
        self.body.get_mut(id)
    }

    pub fn resister(&mut self, nten: Nten) {
        self.body.insert(nten.id, nten);
    }
}

pub struct Context {
    pub varstore: VarStore,
}
impl Context {
    pub fn new(vs: VarStore) -> Self {
        Self { varstore: vs }
    }

    pub fn remove_nten(&mut self, id: &NtenID) -> Result<Nten, String> {
        if let Some(nten) = self.varstore.remove(id) {
            Ok(nten)
        } else {
            Err(format!("Nten id: {} not found in Context", id))
        }
    }
    pub fn insert_nten(&mut self, new_nten: Nten) {
        if let Some(existing_nten) = self.varstore.get_mut(&new_nten.id) {
            LOGGER.debug(format!("nten id: {} is already exists in ctx. value has overrided", new_nten.id));
            self.varstore.body.insert(new_nten.id, new_nten);
        } else {
            self.varstore.body.insert(new_nten.id, new_nten);
        }
    }

    pub fn get_val(&self, id: &NtenID) -> Tensor {
        if let Some(nten) = self.varstore.body.get(id) {
            if let Some(val) = &nten.val {
                // TensorはstoreageがArcでほかがnameとshapeなのでクローンしてよい。
                val.clone()
            } else {
                panic!("val of Nten id: {} is None", id);
            }
        } else {
            panic!("Nten id: {} not found in Context", id);
        }
    }

    pub fn get_grad(&self, id: &NtenID) -> Tensor {
        if let Some(nten) = self.varstore.body.get(id) {
            if let Some(grad) = &nten.grad {
                grad.clone()
            } else {
                panic!("grad of Nten id: {} is None", id);
            }
        } else {
            panic!("Nten id: {} not found in Context", id);
        }
    }

    pub fn insert_val(&mut self, id: &NtenID, tensor: Tensor) {
        if let Some(nten) = self.varstore.get_mut(id) {
            nten.val = Some(tensor);
        } else {
            panic!("Nten id: {} not found in Context", id);
        }
    }

    pub fn add_assign_grad(&mut self, id: &NtenID, new_grad: &Tensor) {
        if let Some(nten) = &mut self.varstore.get_mut(id) {
            if let Some(old_grad) = &nten.grad {
                nten.grad = Some(old_grad.add(&new_grad).unwrap());
            } else {
                nten.grad = Some(new_grad.clone());
            }
        } else {
            panic!("Nten id: {} not found in Context", id)
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
    pub fn new(vs: VarStore) -> Self {
        Self {
            already_executed: HashSet::new(),
            next_execute_index: 0,
            tape: Vec::new(),

            ctx: Context::new(vs),
        }
    }
    // return value
    pub fn step_forward<const N: usize>(&mut self, mut results: [Nten; N]) -> [Nten; N] {
        // グラフ探索してテープを構築，self.tapeに追加される
        self._build_tape(&results);
        // 実行
        for i in self.next_execute_index..self.tape.len() {
            //LOGGER.debug(self.tape[i].name());
            self.tape[i].forward(&mut self.ctx);
            self.already_executed.insert(self.tape[i].get_id());
        }
        // 次回のために開始位置をずらす
        self.next_execute_index = self.tape.len();

        // 結果を詰めて返す
        for nten in results.iter_mut() {
            *nten = self.ctx.varstore.clone_nten(&nten.id);
        }
        results
    }

    pub fn backward<'a>(&'a mut self, result: Nten) -> &'a Context {
        if let None = &result.grad {
            panic!("cannot start backward. grad of result nten is None");
        }
        // loss fn したntenはgradを持っているがctxワールドのntenにはgradがないのでいれる
        self.ctx.insert_nten(result);

        for i in (0..self.tape.len()).rev() {
            //LOGGER.debug(self.tape[i].name());
            self.tape[i].backward(&mut self.ctx);
        }
        &self.ctx
    }

    pub fn _build_tape<const N: usize>(&mut self, results: &[Nten; N]) {
        //! 1, 結果側からグラフ探索を行って結果をテープにする
        //! 2, ContextのVarStoreにすべてのNtenの形だけ作っておく。
        let mut already_seen = HashSet::new();
        let mut stack = Vec::new();
        for tensor in results.iter() {
            stack.push(tensor.creator.clone_box());
        }

        while let Some(node) = stack.pop() {
            // if it was
            if self.already_executed.contains(&node.get_id()) {
                continue;
            }
            if !already_seen.contains(&node.get_id()) {
                already_seen.insert(node.get_id());
                // tape
                self.tape.push(node.clone_box());
                for input in node.sources() {
                    stack.push(input.clone_box());
                }
            }
        }

        self.tape.reverse();
    }
}
