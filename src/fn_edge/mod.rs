
use crate::autograd::Context;

mod add;
pub use add::Add2d;



/*-------------Level 2, graph node & system-----------------------------------
*/
pub trait FnEdge {
    // メンバ変数しかいじらないことで引数がいらなくなり、trait化できる。

    // 計算実行用
    fn forward(&self, ctx: &mut Context);
    fn backward(&self, ctx: &mut Context);
}

// replace it in builder.tape to drop used Tensor in FnEdge if Inference mode
pub struct DummyFnEdge();
impl DummyFnEdge {
    pub fn new() -> Self {
        Self()
    }
}
impl FnEdge for DummyFnEdge {
    fn forward(&self, ctx: &mut Context) {
        panic!("Error: you executed DummyFnEdge.forward()");
    }
    fn backward(&self, ctx: &mut Context) {
        panic!("Error: you executed DummyFnEdge.backward() you may run backward while using Mode::Inference");
    }
}

pub struct HumanCreatedFnEdge();
impl HumanCreatedFnEdge {
    pub fn new() -> Self {
        Self()
    }
}
impl FnEdge for HumanCreatedFnEdge {
    fn forward(&self, ctx: &mut Context) {
        panic!("Error: you executed DummyFnEdge.forward()");
    }
    fn backward(&self, ctx: &mut Context) {
        panic!("Error: you executed DummyFnEdge.backward() you may run backward while using Mode::Inference");
    }
}
