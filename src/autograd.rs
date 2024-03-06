use std::{collections::{HashMap, HashSet}, sync::{Arc, Mutex, MutexGuard}};

use colored::Colorize;
use rayon::result;

use crate::{
    dtype::Dtype, fn_edge::{DummyFnEdge, FnEdge, FnEdgeID}, logger::LOGGER, nten::{Nten, NtenID}, tensor::{Tensor, Tensor2d}
};

#[derive(Clone)]
pub struct VarStore {
    pub(crate) body: Arc<Mutex<HashMap<NtenID, Nten>>>,
    pub parameter_ids: Arc<Mutex<HashSet<NtenID>>>,
    lending: HashSet<NtenID>,
}
impl VarStore {
    pub fn new() -> Self {
        Self {
            body: Arc::new(Mutex::new(HashMap::new())),
            parameter_ids: Arc::new(Mutex::new(HashSet::new())),
            lending: HashSet::new(),
        }
    }

    fn remove_nten(&mut self, id: &NtenID) -> Option<Nten> {
        // 返し忘れるとnot foundのリスク
        if let Some(nten) = self.body.lock().unwrap().remove(id) {
            // 貸出中であることを記録
            self.lending.insert(*id);
            Some(nten)
        } else {
            None
        }
    }
    fn return_nten(&mut self, nten: Nten) {
        self.lending.remove(&nten.id);
        self.body.lock().unwrap().insert(nten.id, nten);
    }

    pub fn resister_parameter(&mut self, nten: Nten) {
        self.parameter_ids.lock().unwrap().insert(nten.id);
        self.body.lock().unwrap().insert(nten.id, nten);
    }
    pub fn resister_input(&mut self, nten: Nten) {
        self.body.lock().unwrap().insert(nten.id, nten);
    }

    pub fn print_all_contents_id(&self) {
        LOGGER.debug(format!("{}::{}() >> now in vs, there are: ", "VarStore".green(), "print_all_contents_id".yellow()));
        if self.body.lock().unwrap().len() == 0 {
            LOGGER.debug(format!("- Nothing in VarStore!"))
        }
        for (id, nten) in self.body.lock().unwrap().iter() {
            LOGGER.debug(format!("- {} id:{} name:\"{}\" shape:{}", "Nten".green(), id, nten.name, nten.shape.to_string()));
        }
        LOGGER.debug(format!("paramter ids are: {:?}", self.parameter_ids));
        LOGGER.debug(format!("lending: {:?}", self.lending.iter()))

    }
}

pub struct Context {
    pub varstore: VarStore,
    temp_tensors: HashMap<NtenID, Tensor>,
}
impl Context {
    pub fn new() -> Self {
        Self {
            varstore: VarStore::new(),
            temp_tensors: HashMap::new(),
        }
    }

    pub fn insert_nten(&mut self, new_nten: Nten) {
        if let Some(existing_nten) = self.varstore.remove_nten(&new_nten.id) {
            LOGGER.debug(format!(
                "nten id: {} is already exists in ctx. value has overrided",
                existing_nten.id
            ));
            self.varstore.body.lock().unwrap().insert(new_nten.id, new_nten);
        } else {
            self.varstore.body.lock().unwrap().insert(new_nten.id, new_nten);
        }
    }

    pub fn get_val(&self, id: &NtenID) -> Tensor {
        if let Some(nten) = self.varstore.body.lock().unwrap().get(id) {
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

    pub fn get_val_as_2d<const R: usize, const C: usize, T: Dtype>(&self, id: &NtenID) -> Tensor2d<R, C, T> {
        if let Some(nten) = self.varstore.body.lock().unwrap().get(id) {
            if let Some(val) = &nten.val {
                // TensorはstoreageがArcでほかがnameとshapeなのでクローンしてよい。
                val.to_typed2d().unwrap()
            } else {
                panic!("val of Nten id: {} is None", id);
            }
        } else {
            LOGGER.error(format!("Context::get_val_as_2d() >> Nten id: {} not found in Context", id));
            panic!()
        }
    }

    pub fn get_grad(&self, id: &NtenID) -> Tensor {
        if let Some(nten) = self.varstore.body.lock().unwrap().get(id) {
            if let Some(grad) = &nten.grad {
                grad.clone()
            } else {
                panic!("grad of Nten id: {} is None", id);
            }
        } else {
            panic!("Nten id: {} not found in Context", id);
        }
    }

    pub fn get_grad_as_2d<const R: usize, const C: usize, T: Dtype>(&self, id: &NtenID) -> Tensor2d<R, C, T> {
        if let Some(nten) = self.varstore.body.lock().unwrap().get(id) {
            if let Some(grad) = &nten.grad {
                grad.clone().to_typed2d().unwrap()
            } else {
                panic!("grad of Nten id: {} is None", id);
            }
        } else {
            panic!("Nten id: {} not found in Context", id);
        }
    }

    pub fn insert_val(&mut self, id: &NtenID, tensor: Tensor) {
        if let Some(mut nten) = self.varstore.remove_nten(id) {
            nten.val = Some(tensor);
            self.varstore.return_nten(nten);
        } else {
            // paramterでもinputでもないランタイムのctxワールドのntenの生成はここで行う。
            let new = Nten {
                id: *id,
                name: "run time insert".to_string(),
                creator: Box::new(DummyFnEdge::new()),
                shape: tensor.shape,
                val: Some(tensor),
                grad: None,
            };
            self.varstore.body.lock().unwrap().insert(*id, new);
        }
    }

    pub fn add_assign_grad(&mut self, id: &NtenID, new_grad: &Tensor) {
        if let Some(mut nten) = self.varstore.remove_nten(id) {
            if let Some(old_grad) = &nten.grad {
                nten.grad = Some(old_grad.add(&new_grad).unwrap());
            } else {
                nten.grad = Some(new_grad.clone());
            }
            self.varstore.return_nten(nten);
        } else {
            panic!("Nten id: {} not found in Context", id)
        }
    }

    pub fn insert_tensor(&mut self, id: &NtenID, tensor: Tensor) {
        self.temp_tensors.insert(*id, tensor);
    }
    pub fn get_tensor(&mut self, id: &NtenID) -> Tensor {
        if let Some(tensor) = self.temp_tensors.get(id) {
            // TensorはstoreageがArcでほかがnameとshapeなのでクローンしてよい。
            tensor.clone()
        } else {
            panic!("temp tensor of Nten id: {} not found in Context", id);
        }
    }
    pub fn get_tensor_as_2d<const R: usize, const C: usize, T: Dtype>(&mut self, id: &NtenID) -> Tensor2d<R, C, T> {
        if let Some(tensor) = self.temp_tensors.get(id) {
            // TensorはstoreageがArcでほかがnameとshapeなのでクローンしてよい。
            tensor.clone().to_typed2d().unwrap()
        } else {
            panic!("temp tensor of Nten id: {} not found in Context", id);
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
    pub fn get_vs(&mut self) -> VarStore {
        self.ctx.varstore.clone()
    }
    // return value
    pub fn step_forward<const N: usize>(&mut self, mut results: [Nten; N]) -> [Nten; N] {

        //self.ctx.varstore.print_all_contents_id();


        // グラフ探索してテープを構築，self.tapeに追加される
        self._build_tape(&results);
        // 実行
        for i in self.next_execute_index..self.tape.len() {
            LOGGER.debug(format!("{}: execute {} of FnEdge id: {}, name: {}", "Autograd".cyan(), "forward".blue(), self.tape[i].get_id(), self.tape[i].name()));
            self.tape[i].forward(&mut self.ctx);
            self.already_executed.insert(self.tape[i].get_id());
        }
        // 次回のために開始位置をずらす
        self.next_execute_index = self.tape.len();

        // 結果を詰めて返す
        for nten in results.iter_mut() {
            let new_nten = self.ctx.varstore.remove_nten(&nten.id).unwrap();
            *nten = new_nten.clone();
            self.ctx.varstore.return_nten(new_nten);
        }
        results
    }

    pub fn backward<'a>(&'a mut self, result: &Nten) -> &'a mut Context {
        let result = result.clone();
        if let None = &result.grad {
            panic!("cannot start backward. grad of result nten is None");
        }
        // loss fn したntenはgradを持っているがctxワールドのntenにはgradがないのでいれる
        self.ctx.insert_nten(result);

        for i in (0..self.tape.len()).rev() {
            LOGGER.debug(format!("{}: execute {} of FnEdge id: {}, name: {}", "Autograd".cyan(), "backward".purple(), self.tape[i].get_id(), self.tape[i].name()));
            self.tape[i].backward(&mut self.ctx);
        }
        &mut self.ctx
    }

    pub fn _build_tape<const N: usize>(&mut self, results: &[Nten; N]) {
        //! 1, 結果側からグラフ探索を行って結果をテープにする
        let mut already_seen = HashSet::new();
        let mut stack = Vec::new();
        for tensor in results.iter() {
            stack.push(tensor.creator.clone_box());
        }

        while let Some(fn_edge) = stack.pop() {
            // when every fn_edge is already executed, stack become empty and end loop
            if self.already_executed.contains(&fn_edge.get_id()) {
                continue;
            }
            // if this is first time to see it
            if !already_seen.contains(&fn_edge.get_id()) {
                already_seen.insert(fn_edge.get_id());

                // tape
                self.tape.push(fn_edge.clone_box());
                for input in fn_edge.sources() {
                    stack.push(input.clone_box());
                }
            }
        }

        self.tape.reverse();
    }

    pub fn zero_grad(&mut self) {
        // 削除すべきキーのリストを作成
        let keys_to_remove: Vec<_> = self.ctx.varstore.body.lock().unwrap()
            .iter()
            .filter_map(|(id, _)| {
                let is_parameter = self.ctx.varstore.parameter_ids.lock().unwrap().contains(id);
                if !is_parameter {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();

        // キーに基づいて要素を削除
        for id in keys_to_remove {
            self.ctx.varstore.body.lock().unwrap().remove(&id);
        }

        // parameterのgradを削除
        for (id, nten) in self.ctx.varstore.body.lock().unwrap().iter_mut() {
            nten.grad = None;
        }
        self.tape.clear();
        self.next_execute_index = 0;
    }
}
