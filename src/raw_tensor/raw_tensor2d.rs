use crate::{backend_cpu::RawDense, dtype::{Dtype, Shape}};

use super::{RawTensor, Storage};

pub struct RawTensor2d<const R: usize, const C: usize, T> {
    pub name: String,
    pub strage: Storage,
    pub dummy: T,
}
impl<const R: usize, const C: usize, T: Dtype> RawTensor2d<R, C, T> {
    pub fn to_untyped(self) -> RawTensor {
        RawTensor {
            name: self.name,
            shape: Shape::D2(R, C),
            strage: self.strage,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            name: "added".to_string(),
            strage: &self.strage + &other.strage,
            dummy: T::default()
        }
    }

    pub fn name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }
}

impl<const R: usize, const C: usize> RawTensor2d<R, C, f32> {
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
            strage: Storage::Densef32(raw_dense),
            dummy: <f32 as Dtype>::default(),
        }
    }
}