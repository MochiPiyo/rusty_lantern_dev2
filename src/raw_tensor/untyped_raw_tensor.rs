use crate::dtype::{Dtype, Shape};

use super::{RawTensor2d, Storage};

pub struct RawTensor {
    pub name: String,
    pub shape: Shape,
    pub strage: Storage,
}
impl RawTensor {
    pub fn to_typed2d<const R: usize, const C: usize, T: Dtype>(self) -> Result<RawTensor2d<R, C, T>, String> {
        if let Shape::D2(r, c) = self.shape {
            if r == R && c == C {
                Ok(RawTensor2d::<R, C, T> {
                    name: self.name,
                    strage: self.strage,
                    dummy: T::default(),
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
}  
