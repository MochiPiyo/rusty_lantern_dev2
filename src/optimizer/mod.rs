
use crate::autograd::Context;

mod sgd;
pub use sgd::Sgd;



pub trait Optimizer {
    fn update(&mut self, ctx: &mut Context);
}

pub(crate) struct DummyOptimizer {
}
impl Optimizer for DummyOptimizer {
    fn update(&mut self, ctx: &mut Context) {
        panic!("you use dummy optimizer");
    }
}