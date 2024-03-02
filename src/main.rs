use std::sync::Arc;

use autograd::{Autograd, VarStore};
use dtype::Dtype;
//use optimizer::Sgd;
use tensor::{Tensor, Tensor2d};
use nten::{Nten, Nten2d};


mod tensor;
mod backend_cpu;
mod fn_edge;
mod nten;
//mod optimizer;
mod autograd;
mod dtype;
mod logger;
mod machine_config;

mod load_mnist;

/*
struct Model {
    parameter: Tensor2d<2, 2, f32>,
}
impl Model {
    fn new() -> Self {
        Self {
            parameter: Nten2d::new_from_martix([
                [5.0, 6.0],
                [7.0, 8.0]
            ]).as_parameter().name("parameter"),
        }
    }
    fn forward(&self, input: &Nten2d<2, 2, f32>) -> Nten2d<2, 2, f32> {
        println!("y");
        let result = self.parameter.add(input);
        println!("y");
        return result;
    }
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
    let input1: Tensor2d<2, 2, f32> = Tensor2d::new_from_martix([
        [1.0, 2.0],
        [3.0, 4.0]
    ]).name("input1");
    let input2: Tensor2d<2, 2, f32> = Tensor2d::new_from_martix([
        [1.0, 2.0],
        [3.0, 4.0]
    ]).name("input2");

    let output = input1.add(&input2);
    println!("{:?}", output.storage);
}

fn dummy_loss_fn(result: &mut Nten) {
    result.grad = result.val.clone();
}

fn nten_add() {
    let input1: Tensor2d<2, 2, f32> = Tensor2d::new_from_martix([
        [1.0, 2.0],
        [3.0, 4.0]
    ]).name("input1");
    let input2: Tensor2d<2, 2, f32> = Tensor2d::new_from_martix([
        [1.0, 2.0],
        [3.0, 4.0]
    ]).name("input2");


    let mut vs = VarStore::new();
    let nten1 = Nten2d::new_from_val(input1).as_parameter(&mut vs);
    println!("{}", nten1.id);
    let nten2 = Nten2d::new_from_val(input2).as_parameter(&mut vs);
    println!("{}", nten2.id);

    let output = nten1.add(&nten2).as_parameter(&mut vs);
    println!("{}", output.id);

    // forwardだけexecute
    let mut autograd = Autograd::new(vs);
    let mut result = autograd.step_forward([output.to_untyped()]);

    println!("result {:?}", result[0].val);

    // backward
    dummy_loss_fn(&mut result[0]);
    let ctx = autograd.backward(result[0].clone());
    
    println!("grad {:?}", ctx.get_grad(&nten1.id))
    /*
    1
    2
    3
    id: 3
    id: 2
    id: 1
    you executed DummyFnEdge.forward()
    you executed DummyFnEdge.forward()
    Add2d<2,2,f32> forward
    Tensor2d { name: "added", storage: RwLock { data: RawData::Densef32(RawDense { body: [2.0, 4.0, 6.0, 8.0] }), poisoned: false, .. }, _marker: PhantomData<f32> }
    result Some(Tensor { name: "added", shape: D2(2, 2), storage: RwLock { data: RawData::Densef32(RawDense { body: [2.0, 4.0, 6.0, 8.0] }), poisoned: false, .. } })
    nten id: 3 is already exists in ctx. value has overrided
    Add2d<2,2,f32> backward
    you executed DummyFnEdge.backward() you may run backward while using Mode::Inference
    you executed DummyFnEdge.backward() you may run backward while using Mode::Inference
    grad Tensor { name: "added", shape: D2(2, 2), storage: RwLock { data: RawData::Densef32(RawDense { body: [2.0, 4.0, 6.0, 8.0] }), poisoned: false, .. } }
        
     */
}


struct Linear<const N: usize, const M: usize> {
    pub parameter: Nten2d<N, M, f32>,
}
impl<const N: usize, const M: usize> Linear<N, M> {
    fn new(vs: &mut VarStore, val: Tensor2d<N, M, f32>) -> Self {
        Self {
            parameter: Nten2d::new_from_val(val).as_parameter(vs)
        }
    }
    fn forward<const I: usize>(&self, input: &Nten2d<I, N, f32>, vs: &mut VarStore) -> Nten2d<I, M, f32> {
        let result = nten::matmul(input, &self.parameter).as_parameter(vs);
        result
    }
}
fn all_one_loss_fn(result: &mut Nten) {
    result.grad = Some(Tensor::new_ones::<f32>(result.shape));
}
fn linear() {
    let input_val: Tensor2d<2, 2, f32> = Tensor2d::new_from_martix([
        [1.0, 2.0],
        [3.0, 4.0]
    ]).name("input1");

    let mut vs = VarStore::new();
    let input = Nten2d::new_from_val(input_val).as_parameter(&mut vs);
    println!("input id {}", input.id);

    let parameter: Tensor2d<2, 3, f32> = Tensor2d::new_from_martix([
        [1.0, 2.0, 3.0],
        [3.0, 4.0, 5.0]
    ]).name("parameter");
    let linear: Linear<2, 3> = Linear::new(&mut vs, parameter);

    let result = linear.forward(&input, &mut vs);

    let mut autograd = Autograd::new(vs);
    let mut result = autograd.step_forward([result.to_untyped()]);
    println!("{:?}", result[0].val);
    /* 外部で計算した正解の値
    7	10	13
    15	22	29
    */
    
    all_one_loss_fn(&mut result[0]);
    let ctx = autograd.backward(result[0].clone());

    let param_diff = ctx.get_grad(&linear.parameter.id);
    println!("{:?}", param_diff);
    /* numpyで計算した正解
    https://colab.research.google.com/drive/1CONgJAvPqw_jIqrBkaCTjObwzeQvNUQx#scrollTo=vrAtJ1xUx0He
    [[4. 4. 4.]
     [6. 6. 6.]]
     */
    let input_diff = ctx.get_grad(&input.id);
    println!("{:?}", input_diff);
    /* numpyで計算した正解
    [[ 6. 12.]
     [ 6. 12.]]
     */
}


struct Mnist {

}


fn main() {
    //raw_add();
    //nten_add();
    linear();

}
