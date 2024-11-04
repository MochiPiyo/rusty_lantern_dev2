use indicatif::ProgressBar;

use crate::{autograd::{Autograd, Context, VarStore}, lantern_datasets, loss_fn, nten::{self, Nten2d}, optimizer::{Optimizer, Sgd}, tensor::Tensor2d};




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
pub fn mnist() {
    const BATCH_SIZE: usize = 512;
    const HIDDEN_SIZE: usize = 256;
    let learning_rate: f32 = 0.01;
    let num_epoch: usize = 100;
    let print_interval: usize = 10;

    // load dataset
    // gzは解凍されていること
    let train_image_path = "./mnist_data/train-images.idx3-ubyte";
    let train_label_path = "./mnist_data/train-labels.idx1-ubyte";
    let (train_images, train_labels): (Vec<[[u8; 28]; 28]>, Vec<u8>)
         = lantern_datasets::load_minst(train_image_path, train_label_path);
    let train_images: Vec<[u8; 784]> = lantern_datasets::selialize_minst(&train_images);

    // tools for learning
    // Autogradは構築した計算グラフを最適化・実行する
    let mut autograd = Autograd::new();
    // VarStoreはパラメーターと入力変数を格納する
    let mut vs = autograd.get_vs();
    // 
    let mut optimizer = Sgd::new(learning_rate);

    // create model
    let model: Model<BATCH_SIZE, HIDDEN_SIZE> = Model::new(&mut vs);

    let progress_bar = ProgressBar::new(print_interval as u64);
    println!("start learning");
    for epoch in 0..num_epoch {
        // shuffle and make batch
        let (train_image_batches,
            train_label_batches): (Vec<Tensor2d<BATCH_SIZE, 784, f32>>, Vec<Tensor2d<BATCH_SIZE, 10, f32>>)
             = lantern_datasets::shuffle_and_make_batch(&train_images, &train_labels);
        
        // learn batch
        for (i, (images, labels)) in train_image_batches.iter().zip(train_label_batches.iter()).enumerate() {
            progress_bar.set_position((i % print_interval) as u64);
            
            // mark as input !
            let images = Nten2d::new_from_val(images.clone())
                .name("input")
                .as_input(&mut vs);

            
            let graph = model.forward(&images);
           
            vs.print_all_contents_id();
            let mut predict = autograd.step_forward([graph.to_untyped()]);
            
            let loss = loss_fn::softmax_cross_entropy_f32(&mut predict[0], labels.to_untyped());

            let ctx: &mut Context = autograd.backward(&predict[0]);
            
            // update parameter
            optimizer.update(ctx);
            autograd.zero_grad();

            // epochの最後のバッチについて正解率と損失を出力
            if i % print_interval == 0 {
                let predict_index = predict[0].val.clone().unwrap().top_index_per_batch();
                let label_index = labels.top_index_per_batch();
                
                let mut acc = 0;
                for (p, l) in predict_index.iter().zip(label_index.iter()) {
                    if *p == *l {
                        acc += 1;
                    }
                }
                println!("epoch {} loop {}, Loss: {}, acc: {:.2}%", epoch+1, i, loss, acc as f32/BATCH_SIZE as f32 * 100.0);
            }
        }
    }
}