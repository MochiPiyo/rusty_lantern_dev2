use std::sync::RwLockWriteGuard;

use crate::{backend_cpu::RawDense, dtype::{Dtype, Shape}, nten::Nten, tensor::{Storage, Tensor}};

// implaceは危険なので制限する
impl Tensor {
    pub(in crate::loss_fn) fn storage_mut(&self) -> RwLockWriteGuard<'_, Storage> {
        self.storage.write().unwrap()
    }
}



fn softmax_per_batch(input: &mut Tensor) {}

// 
pub fn softmax_cross_entropy_f32(predict: &mut Nten, teacher: Tensor) -> f32 {
    // Assuming the logits are stored in predict.val and the labels are stored in teacher
    let logits = if let Storage::Densef32(ref raw_dense) =
        &*predict.val.unwrap().storage()
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

    for i in 0..batch_size {
        let start_idx = i * one_data_len;
        let end_idx = start_idx + one_data_len;
        let logits_slice = &logits[start_idx..end_idx];
        let labels_slice = &labels[start_idx..end_idx];

        // Softmax
        let max_logit = logits_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits_slice.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp_logits: f32 = exp_logits.iter().sum();
        let softmax: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp_logits).collect();

        // Cross-entropy loss
        let log_softmax: Vec<f32> = softmax.iter().map(|&x| x.ln()).collect();
        for (s, l) in log_softmax.iter().zip(labels_slice.iter()) {
            loss -= (l * s) as f64;
        }

        // Gradient of the loss w.r.t. the logits
        for (g, (&s, &l)) in grad[start_idx..end_idx].iter_mut().zip(softmax.iter().zip(labels_slice.iter())) {
            *g += (s - l) / batch_size as f32;
        }
    }

    loss = loss / batch_size as f64;

    // Update the gradient in the predict tensor
    *predict.grad.unwrap().storage_mut() = Storage::Densef32(RawDense { body: grad });

    loss as f32
}
