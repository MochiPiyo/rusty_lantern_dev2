//use autograd::Autograd;
//use optimizer::Sgd;
use raw_tensor::RawTensor2d;
//use tensor::Tensor2d;


mod raw_tensor;
mod backend_cpu;
//mod fn_edge;
//mod tensor;
//mod optimizer;
//mod autograd;
mod dtype;
mod logger;
mod machine_config;


/*
struct Model {
    parameter: Tensor2d<2, 2, f32>,
}
impl Model {
    fn new() -> Self {
        Self {
            parameter: Tensor2d::new_from_martix([
                [5.0, 6.0],
                [7.0, 8.0]
            ]).as_parameter().name("parameter"),
        }
    }
    fn forward(&self, input: &Tensor2d<2, 2, f32>) -> Tensor2d<2, 2, f32> {
        println!("y");
        let result = self.parameter.add(input);
        println!("y");
        return result;
    }
}

fn dummy_loss_fn<const R: usize, const C: usize, T>(result: &mut Tensor2d<R, C, T>) {
    result.grad = result.val;
}

fn add_grad() {
    let model = Model::new();

    let autograd = Autograd::new();
    let lr = 0.01;
    let optimizer = Sgd::new(lr);
    

    for _ in 0..1 {
        let input: Tensor2d<2, 2, f32> = Tensor2d::new_from_martix([
            [1.0, 2.0],
            [3.0, 4.0]
        ]).name("input");

        let graph = model.forward(&input);
        let mut result = autograd.step_forward(vec![graph.to_untyped()]).name("result");
        dummy_loss_fn(&mut result);

        let context = autograd.backward(result);
        optimizer.update(context);
    }
}
*/

fn raw_add() {
    let input1: RawTensor2d<2, 2, f32> = RawTensor2d::new_from_martix([
        [1.0, 2.0],
        [3.0, 4.0]
    ]).name("input1");
    let input2: RawTensor2d<2, 2, f32> = RawTensor2d::new_from_martix([
        [1.0, 2.0],
        [3.0, 4.0]
    ]).name("input2");

    let output = input1.add(&input2);
    println!("{:?}", output.strage);
}

fn main() {
    raw_add();
}
