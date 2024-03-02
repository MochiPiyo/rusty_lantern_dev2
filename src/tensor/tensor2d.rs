use std::{fmt::Debug, marker::PhantomData, sync::{Arc, RwLock}};

use crate::{backend_cpu::RawDense, dtype::{Dtype, Shape}};

use super::{Tensor, Storage};

#[derive(Debug)]
pub struct Tensor2d<const R: usize, const C: usize, T> {
    pub name: String,
    pub storage: Arc<Storage>,
    pub _marker: PhantomData<T>,
}

impl<const R: usize, const C: usize, T: Dtype> Tensor2d<R, C, T> {
    pub fn storage(&self) -> Arc<Storage> {
        self.storage.clone()
    }

    pub fn to_untyped(self) -> Tensor {
        Tensor {
            name: self.name,
            shape: Shape::D2(R, C),
            storage: self.storage,
        }
    }

    pub fn name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            name: "added".to_string(),
            // &は演算で所有権を消費しないため，＊はRwLockGuardの参照をとるため
            storage: Arc::new(&*self.storage() + &*other.storage()),
            _marker: PhantomData,
        }
    }

    pub fn transpose(&self) -> Tensor2d<C, R, T> {
        // <R, C, T> to <C, R, T>
        Tensor2d::<C, R, T> {
            name: format!("{} -> transposed", self.name),
            storage: Arc::new(self.storage().transpose(Shape::D2(R, C))),
            _marker: PhantomData,
        }
    }
}


impl<const R: usize, const C: usize> Tensor2d<R, C, f32> {
    pub fn new_from_martix(matrix: [[f32; C]; R]) -> Self {
        let mut data = Vec::with_capacity(R*C);
        for i in 0..matrix.len() {
            data.extend_from_slice(&matrix[i]);
        }
        let raw_dense = RawDense {
            body: data,
        };
        Self {
            name: "no_name".to_string(),
            storage: Arc::new(Storage::Densef32(raw_dense)),
            _marker: PhantomData,
        }
    }

    pub fn new_from_vec(data: Vec<f32>) -> Result<Self, ()> {
        if data.len() != R * C {
            return Err(());
        }
        Ok(Self {
            name: String::new(),
            storage: Arc::new(Storage::Densef32(RawDense { body: data })),
            _marker: PhantomData,
        })
    }

    pub fn new_ones() -> Self {
        let mut body = Vec::new();
            for i in 0..R {
                for j in 0..C {
                    body.push(1.0);
                }
            }
        let raw_dense = RawDense { body };
        Self {
            name: "no_name".to_string(),
            storage: Arc::new(Storage::Densef32(raw_dense)),
            _marker: PhantomData,
        }
    }
}