use std::{marker::PhantomData, sync::{Arc, RwLock}};

use crate::{backend_cpu::RawDense, dtype::{Dtype, Shape}};

use super::{Tensor2d, Storage};

#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub shape: Shape,
    // dtype
    pub storage: Arc<Storage>,
}
impl Tensor {
    pub fn new_empty() -> Self {
        Self {
            name: "created by Tensor::new_empty".to_string(),
            shape: Shape::D1(0),
            storage: Arc::new(Storage::None),
        }
    }

    pub fn new_ones<T: Dtype>(shape: Shape) -> Self {
        match shape {
            Shape::D1(n) => Self {
                name: "ones".to_string(),
                shape,
                storage: Arc::new(Storage::Densef32(RawDense { body: vec![1.0_f32;n] }))
            },
            Shape::D2(n, m) => Self {
                name: "ones".to_string(),
                shape,
                storage: Arc::new(Storage::Densef32(RawDense { body: vec![1.0_f32;n * m] }))
            }
        }
    }

    pub fn storage(&self) -> Arc<Storage> {
        self.storage.clone()
    }

    pub fn to_typed2d<const R: usize, const C: usize, T: Dtype>(&mut self) -> Result<Tensor2d<R, C, T>, String> {
        if let Shape::D2(r, c) = self.shape {
            if r == R && c == C {
                Ok(Tensor2d::<R, C, T> {
                    name: self.name.clone(),
                    storage: self.storage.clone(),
                    _marker: PhantomData,
                })
            } else {
                Err(format!("RawTensor cast error: expected Shape::D2({}, {}), found Shape::D2({}, {})", R, C, r, c))
            }
        } else {
            Err(format!("RawTensor cast error: expected Shape::D2({}, {}), found {}", R, C, self.shape.to_string()))
        }
    }

    pub fn name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    pub fn add(&self, other: &Self) -> Result<Self, String> {
        if self.shape != other.shape {
            return Err(format!("self. shape {}, other shape {}", self.shape, other.shape));
        }
        Ok(Self {
            name: "added".to_string(),
            shape: self.shape.clone(),
            storage: Arc::new(&*self.storage() + &*other.storage()),
        })
    }
}  
