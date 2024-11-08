use std::{process::exit, sync::Arc};

use autograd::{Autograd, VarStore};
use colored::Colorize;
use dtype::Dtype;
use lantern_datasets::{load_minst, shuffle_and_make_batch};
//use optimizer::Sgd;
use tensor::{Tensor, Tensor2d};
use nten::{Nten, Nten2d};

use crate::{autograd::Context, lantern_datasets::selialize_minst, optimizer::{Optimizer, Sgd}};


mod tensor;
mod backend_cpu;
mod fn_edge;
mod loss_fn;
mod nten;
mod optimizer;
mod autograd;
mod dtype;
mod logger;
mod machine_config;

mod lantern_datasets;
mod example;

/*
MNISTの学習デモは./example.rsを見てください。



main.rsでは開発過程で少しずつ動作検証をした過程を残してあります。

fn raw_add()
単純な足し算です。lanternでは様々な数値型，GPU，疎行列等に対応するため
抽象化レイヤーが入っているので，ここをまず動くことを確かめます。

fn nten_add()
自動微分可能な行列形式Ntenのテストおよび，自動微分機構のAutogradの動作テストです。

fn matmul()
行列積の自動微分です。

fn mnist()
デバッグ用なのでMNISTの学習デモは./example.rsを見てください。
実際のデータセットを使って学習ができることを示しました。ここでは，データセットの作成，
NNレイヤーの構築，学習のループとモデルパラメーターの更新，結果の評価を行いました。
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


    let mut autograd = Autograd::new();
    let mut vs = autograd.get_vs();
    let nten1 = Nten2d::new_from_val(input1).as_parameter(&mut vs);
    println!("{}", nten1.id);
    let nten2 = Nten2d::new_from_val(input2).as_parameter(&mut vs);
    println!("{}", nten2.id);

    let output = nten1.add(&nten2).as_parameter(&mut vs);
    println!("{}", output.id);

    // forwardだけexecute
    let mut result = autograd.step_forward([output.to_untyped()]);

    println!("result {:?}", result[0].val);

    // backward
    dummy_loss_fn(&mut result[0]);
    let ctx = autograd.backward(&result[0]);
    
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


struct Matmul<const N: usize, const M: usize> {
    pub parameter: Nten2d<N, M, f32>,
}
impl<const N: usize, const M: usize> Matmul<N, M> {
    fn new(vs: &mut VarStore, val: Tensor2d<N, M, f32>) -> Self {
        Self {
            parameter: Nten2d::new_from_val(val).name("parameter").as_parameter(vs)
        }
    }
    fn forward<const I: usize>(&self, input: &Nten2d<I, N, f32>) -> Nten2d<I, M, f32> {
        let result = nten::matmul(input, &self.parameter);
        result
    }
}
fn all_one_loss_fn(result: &mut Nten) {
    result.grad = Some(Tensor::new_ones::<f32>(result.shape));
}
fn matmul() {
    
    let mut autograd = Autograd::new();
    let mut vs = autograd.get_vs();

    // create input
    let input_val: Tensor2d<2, 2, f32> = Tensor2d::new_from_martix([
        [1.0, 2.0],
        [3.0, 4.0]
    ]);
    let input = Nten2d::new_from_val(input_val).name("input").as_input(&mut vs);
    println!("input id {}", input.id);

    // create matmul
    let parameter_val: Tensor2d<2, 3, f32> = Tensor2d::new_from_martix([
        [1.0, 2.0, 3.0],
        [3.0, 4.0, 5.0]
    ]);
    let linear: Matmul<2, 3> = Matmul::new(&mut vs, parameter_val);

    // build graph
    let result = linear.forward(&input);
    

    vs.print_all_contents_id();
    let mut result = autograd.step_forward([result.to_untyped()]);
    println!("{:?}", result[0].val);
    /* 外部で計算した正解の値
    7	10	13
    15	22	29
    */
    
    all_one_loss_fn(&mut result[0]);
    let ctx = autograd.backward(&result[0]);

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


struct Linear<const I: usize, const O: usize> {
    weight: Nten2d<I, O, f32>,
    bias: Nten2d<1, O, f32>,
}
impl<const I: usize, const O: usize> Linear<I, O> {
    fn new(vs: &mut VarStore) -> Self {
        // pytorchと同じ初期化
        // U(-\sqrt{k}, \sqrt{k}), k = 1 / 入力特徴数
        let k: f32 = 1.0 / I as f32;
        let weight: Tensor2d<I, O, f32> = Tensor2d::new_uniform(-k.sqrt(), k.sqrt());
        //println!("weight @new {:?}", weight);
        //let weight: Tensor2d<I, O, f32> = Tensor2d::new_init_he();
        let bias: Tensor2d<1, O, f32> = Tensor2d::new_zeros();
        Self {
            weight: Nten2d::new_from_val(weight).name("Linear weight").as_parameter(vs),
            bias: Nten2d::new_from_val(bias).name("Linear bias").as_parameter(vs),
        }
    }
    fn forward<const B: usize>(&self, input: &Nten2d<B, I, f32>) -> Nten2d<B, O, f32> {
        // グラフ構築層
        let out: Nten2d<B, O, f32> = nten::matmul(&input, &self.weight);
        // batch first
        out.add_broadcast(&self.bias)
    }
}

// B: batch, H: hidden
struct Model<const B: usize, const H: usize> {
    linear1: Linear<784, H>,
    linear2: Linear<H, 10>,
}
impl<const B: usize, const H: usize> Model<B, H> {
    fn new(vs: &mut VarStore) -> Self {
        Self {
            linear1: Linear::new(vs),
            linear2: Linear::new(vs),
        }
    }
    fn forward(&self, input: &Nten2d<B, 784, f32>) -> Nten2d<B, 10, f32> {
        // layer層の操作
        let x: Nten2d<B, H, f32> = self.linear1.forward(input);
        let x: Nten2d<B, H, f32> = x.relu();
        let x: Nten2d<B, 10, f32> = self.linear2.forward(&x);
        // return x
        x
    }
}
fn mnist() {
    const BATCH_SIZE: usize = 500;
    const HIDDEN_SIZE: usize = 300;
    let learning_rate: f32 = 0.01;
    let num_epoch: usize = 50000;

    // load dataset
    // gzは解凍されていること
    let train_image_path = "./mnist_data/train-images.idx3-ubyte";
    let train_label_path = "./mnist_data/train-labels.idx1-ubyte";
    let (train_images, train_labels): (Vec<[[u8; 28]; 28]>, Vec<u8>)
         = load_minst(train_image_path, train_label_path);
    let train_images: Vec<[u8; 784]> = selialize_minst(&train_images);

    // tools for learning
    let mut autograd = Autograd::new();
    let mut vs = autograd.get_vs();
    let mut optimizer = Sgd::new(learning_rate);

    // create model
    let model: Model<BATCH_SIZE, HIDDEN_SIZE> = Model::new(&mut vs);

    let mut losses: Vec<f32> = Vec::new();
    for i in 0..num_epoch {
        //println!("{:?}", model.linear2.weight.val);
        //println!("{:?}", model.linear2.bias.val);

        // shuffle and make batch
        let (train_image_batches,
            train_label_batches): (Vec<Tensor2d<BATCH_SIZE, 784, f32>>, Vec<Tensor2d<BATCH_SIZE, 10, f32>>)
             = shuffle_and_make_batch(&train_images, &train_labels);
        
        let mut loss = 0.0;
        // learn batch
        for (e, (images, labels)) in train_image_batches.iter().zip(train_label_batches.iter()).enumerate() {
            // mark as input !
            let images = Nten2d::new_from_val(images.clone())
                .name("input")
                .as_input(&mut vs);

            //println!("weight @ before graph build {:?}", model.linear1.weight.val.as_ref().unwrap());
            let graph = model.forward(&images);
            //println!("weight @ after graph build {:?}", model.linear1.weight.val.as_ref().unwrap());
            vs.print_all_contents_id();
            let mut predict = autograd.step_forward([graph.to_untyped()]);
            //println!("weight @ after execute forward {:?}", model.linear1.weight.val.as_ref().unwrap());

            //println!("{}, {:?}", "predict".red(), predict[0].val.as_ref().unwrap());

            loss = loss_fn::softmax_cross_entropy_f32(&mut predict[0], labels.to_untyped());

            let ctx: &mut Context = autograd.backward(&predict[0]);
            
            if e == 0 {
                let predict_index = predict[0].val.clone().unwrap().top_index_per_batch();
                let label_index = labels.top_index_per_batch();
                //println!("{:?}\n{:?}", predict_index, label_index);
                let mut acc = 0;
                for (p, l) in predict_index.iter().zip(label_index.iter()) {
                    if *p == *l {
                        acc += 1;
                    }
                }
                println!("acc {:.2}%", acc as f32/BATCH_SIZE as f32 * 100.0);

                //println!("linear1.weight val {:?}", ctx.get_val(&model.linear1.weight.id));
                //println!("linear1.bias val {:?}", ctx.get_val(&model.linear1.bias.id));
                //println!("linear2.weight val {:?}", ctx.get_val(&model.linear2.weight.id));
                //println!("linear2.bias val {:?}", ctx.get_val(&model.linear2.bias.id));

                //println!("linear1.weight grad {:?}", ctx.get_grad(&model.linear1.weight.id));
                //println!("linear1.bias grad {:?}", ctx.get_grad(&model.linear1.bias.id));
                //println!("linear2.weight grad {:?}", ctx.get_grad(&model.linear2.weight.id));
                //println!("linear2.bias grad {:?}", ctx.get_grad(&model.linear2.bias.id));

            }
            // update parameter
            optimizer.update(ctx);
            autograd.zero_grad();

            //println!("{:?}", model.linear2.weight.val);
            //println!("{:?}", model.linear2.bias.val);
            //exit(0);
        }
        //println!("{:?}", model.linear2.bias.val);
        println!("epoch {}, Loss is {}", i, loss);
        losses.push(loss);
    }

    println!("Losses: {:?}", losses);
}




fn main() {
    //raw_add();
    //nten_add();
    //matmul();
    //mnist();

    example::mnist()

}
