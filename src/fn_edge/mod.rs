
use std::fmt;

use crate::autograd::Context;

mod add;
pub use add::Add2d;



/*-------------Level 2, graph node & system-----------------------------------
*/
pub trait FnEdge {
    // 計算グラフ構築用
    fn id(&self) -> FnEdgeID;
    fn inputs(&self) -> Vec<Box<dyn FnEdge>>;
    fn clone_box(&self) -> Box<dyn FnEdge>;
    // 計算実行用
    fn forward(&self, ctx: &mut Context);
    fn backward(&self, ctx: &mut Context);

}
// Implement Clone for Box<dyn FnEdge> using the clone_box method
impl Clone for Box<dyn FnEdge> {
    fn clone(&self) -> Box<dyn FnEdge> {
        self.clone_box()
    }
}

#[derive(Eq, Hash, PartialEq, Clone, Copy)]
pub struct FnEdgeID(pub u32);
pub fn get_new_fn_edge_id() -> FnEdgeID {
    use std::sync::atomic;
    static COUNTER: atomic::AtomicU32 = atomic::AtomicU32::new(1);
    // get unique id
    let u32 = COUNTER.fetch_add(1, atomic::Ordering::Relaxed);
    FnEdgeID(u32)
}
impl fmt::Display for FnEdgeID {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// replace it in builder.tape to drop used Tensor in FnEdge if Inference mode
#[derive(Clone)]
pub struct DummyFnEdge {
    id: FnEdgeID,
}
impl DummyFnEdge {
    pub fn new() -> Self {
        Self {
            id: get_new_fn_edge_id(),
        }
    }
}
impl FnEdge for DummyFnEdge {
    fn id(&self) -> FnEdgeID {
        self.id
    }
    fn inputs(&self) -> Vec<Box<dyn FnEdge>> {
        // this must return vec of nothing for stop graph walk in making tape stage
        vec![]
    }
    fn forward(&self, ctx: &mut Context) {
        panic!("Error: you executed DummyFnEdge.forward()");
    }
    fn backward(&self, ctx: &mut Context) {
        panic!("Error: you executed DummyFnEdge.backward() you may run backward while using Mode::Inference");
    }
    fn clone_box(&self) -> Box<dyn FnEdge> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct HumanCreatedFnEdge {
    id: FnEdgeID,
}
impl HumanCreatedFnEdge {
    pub fn new() -> Self {
        Self {
            id: get_new_fn_edge_id(),
        }
    }
}
impl FnEdge for HumanCreatedFnEdge {
    fn id(&self) -> FnEdgeID {
        self.id
    }
    fn inputs(&self) -> Vec<Box<dyn FnEdge>> {
        vec![]
    }
    fn forward(&self, ctx: &mut Context) {
        panic!("Error: you executed DummyFnEdge.forward()");
    }
    fn backward(&self, ctx: &mut Context) {
        panic!("Error: you executed DummyFnEdge.backward() you may run backward while using Mode::Inference");
    }
    fn clone_box(&self) -> Box<dyn FnEdge> {
        Box::new(self.clone())
    }
}
