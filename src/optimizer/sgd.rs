use super::Optimizer;


pub struct Sgd {
    learning_rate: f32,
}
impl Sgd {
    // new is separated from Optimizer trait. because you may pass some different initial arguments
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
        }
    }
}
impl Optimizer for Sgd {
    fn update(&mut self, ctx: &mut crate::autograd::Context) {
        for parameter_id in ctx.varstore.parameter_ids.iter() {
            let grad = ctx.get_grad(parameter_id);
            grad.mul_scalar(self.learning_rate);

            let val = ctx.get_val(parameter_id);

        }
    }
}