use crate::dtype::{Dtype, Shape};

use super::{RawTensor2d, Storage};

#[derive(Clone)]
pub struct RawTensor {
    pub name: String,
    pub shape: Shape,
    pub storage: Storage,
}
impl RawTensor {
    pub fn to_typed2d<const R: usize, const C: usize, T: Dtype>(self) -> Result<RawTensor2d<R, C, T>, String> {
        if let Shape::D2(r, c) = self.shape {
            if r == R && c == C {
                Ok(RawTensor2d::<R, C, T> {
                    name: self.name,
                    storage: self.storage,
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

    pub fn add(&self, other: &Self) -> Result<Self, String> {
        if self.shape != other.shape {
            return Err(format!("self. shape {}, other shape {}", self.shape, other.shape));
        }
        Ok(Self {
            name: "added".to_string(),
            shape: self.shape.clone(),
            storage: &self.storage + &other.storage,
        })
    }
}  
