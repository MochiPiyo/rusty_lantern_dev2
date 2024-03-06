
use std::fmt;

use crate::{autograd::Context, nten::NtenID};

// reference impl
mod add;
pub use add::Add2d;


mod add_broadcst;
pub use add_broadcst::AddBroadcast2d;
mod matmul;
pub use matmul::Matmul;
pub mod relu;
pub use relu::Relu2d;



/*-------------Level 2, graph node & system-----------------------------------
*/
pub trait FnEdge {
    //fn name(&self) -> String;
    // 計算グラフ構築用
    fn get_id(&self) -> FnEdgeID;
    fn sources(&self) -> Vec<Box<dyn FnEdge>>;
    fn inputs(&self) -> Vec<NtenID>; // ctxにNtenをビルド時に予め作成しておくため
    fn clone_box(&self) -> Box<dyn FnEdge>;
    // 計算実行用
    fn forward(&self, ctx: &mut Context);
    fn backward(&self, ctx: &mut Context);

    // デバック
    //fn name(&self) -> String;
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
    fn get_id(&self) -> FnEdgeID {
        self.id
    }
    fn sources(&self) -> Vec<Box<dyn FnEdge>> {
        // this must return vec of nothing for stop graph walk in making tape stage
        vec![]
    }
    fn inputs(&self) -> Vec<NtenID> {
        vec![]
    }
    fn forward(&self, ctx: &mut Context) {
        println!("you executed DummyFnEdge.forward()");
    }
    fn backward(&self, ctx: &mut Context) {
        println!("you executed DummyFnEdge.backward() you may run backward while using Mode::Inference");
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
    fn get_id(&self) -> FnEdgeID {
        self.id
    }
    fn sources(&self) -> Vec<Box<dyn FnEdge>> {
        vec![]
    }
    fn inputs(&self) -> Vec<NtenID> {
        vec![]
    }
    fn forward(&self, ctx: &mut Context) {
        println!("you executed DummyFnEdge.forward()");
    }
    fn backward(&self, ctx: &mut Context) {
        println!("you executed DummyFnEdge.backward() you may run backward while using Mode::Inference");
    }
    fn clone_box(&self) -> Box<dyn FnEdge> {
        Box::new(self.clone())
    }
}
