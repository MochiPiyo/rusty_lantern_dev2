use std::sync::RwLockWriteGuard;

use colored::Colorize;

use crate::{backend_cpu::RawDense, dtype::{Dtype, Shape}, nten::Nten, tensor::{Storage, Tensor, Tensor2d}};

// implaceは危険なので制限する
impl Tensor {
    pub(in crate::loss_fn) fn storage_mut(&self) -> RwLockWriteGuard<'_, Storage> {
        self.storage.write().unwrap()
    }
}

// set predict.grad and returns loss of this batch
pub fn softmax_cross_entropy_f32(predict: &mut Nten, teacher: Tensor) -> f32 {
    // Assuming the logits are stored in predict.val and the labels are stored in teacher
    let logits = if let Storage::Densef32(ref raw_dense) =
        *predict.val.clone().unwrap().storage()
    {
        raw_dense.body.clone()
    } else {
        panic!("softmax_cross_entropy_f32 >> Predict tensor is not of type Densef32");
    };

    let labels = if let Storage::Densef32(ref raw_dense) = *teacher.storage() {
        raw_dense.body.clone()
    } else {
        panic!("softmax_cross_entropy_f32 >> Teacher tensor is not of type Densef32");
    };

    if predict.shape != teacher.shape {
        panic!("softmax_cross_entropy_f32>> predict Shape is {} but teacher Shape is {}", predict.shape.to_string(), teacher.shape.to_string());
    }
    let one_data_len = match predict.shape {
        Shape::D1(n) => n,
        Shape::D2(_r, c) => c,
    };


    let batch_size = logits.len() / one_data_len;
    let mut loss: f64 = 0.0;
    let mut grad = vec![0.0; logits.len()];

    //println!("change batch");
    for i in 0..batch_size {
        let start_idx = i * one_data_len;
        let end_idx = start_idx + one_data_len;
        //println!("{}", start_idx);
        let logits_slice = &logits[start_idx..end_idx];
        let labels_slice = &labels[start_idx..end_idx];
        //println!("logits_slice{:?}", logits_slice);
        // Softmax
        // オーバーフロー防止のためlargestを全体から引く
        let max_logit = logits_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits_slice.iter().map(|&x| (x - max_logit).exp()).collect();
        //println!("exp_logits{:?}", exp_logits);
        // exp/exp_sum
        let sum_exp_logits: f32 = exp_logits.iter().sum();
        let softmax: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp_logits).collect();
        //println!("softmax{:?}", softmax);
        // Cross-entropy loss
        // ! f32::MINを使ってはいけない
        let log_softmax: Vec<f32> = softmax.iter().map(|&x| (x + f32::EPSILON).ln()).collect();
        //println!("log_softmax{:?}", log_softmax);
        for (s, l) in log_softmax.iter().zip(labels_slice.iter()) {
            //print!("({}, {})", s, l);
            loss -= (l * s) as f64 / batch_size as f64;
        }

        // Gradient of the loss w.r.t. the logits
        for (g, (&s, &l)) in grad[start_idx..end_idx].iter_mut().zip(softmax.iter().zip(labels_slice.iter())) {
            *g += (s - l) / batch_size as f32;
            //*g = s - l;
        }
    }
    // バッチサイズで割る
    //let _ = grad.iter_mut().map(|i| *i / batch_size as f32);

    //print!("{}: {:.5?}", "loss fn grad".red(), grad);

    // Update the gradient in the predict tensor
    predict.set_grad(Tensor::new_from_vec(grad, predict.shape).unwrap());
    
    loss as f32
}
