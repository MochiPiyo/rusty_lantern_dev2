use std::{marker::PhantomData, sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard}};

use crate::{backend_cpu::RawDense, dtype::{Dtype, Shape}, logger::LOGGER, main};

use super::{storage, Storage, Tensor2d};

#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub shape: Shape,
    // dtype
    pub storage: Arc<RwLock<Storage>>,
}
impl Tensor {
    pub fn new_empty() -> Self {
        Self {
            name: "created by Tensor::new_empty".to_string(),
            shape: Shape::D1(0),
            storage: Arc::new(RwLock::new(Storage::None)),
        }
    }

    pub fn new_ones<T: Dtype>(shape: Shape) -> Self {
        match shape {
            Shape::D1(n) => Self {
                name: "ones".to_string(),
                shape,
                storage: Arc::new(RwLock::new(Storage::Densef32(RawDense { body: vec![1.0_f32;n] })))
            },
            Shape::D2(n, m) => Self {
                name: "ones".to_string(),
                shape,
                storage: Arc::new(RwLock::new(Storage::Densef32(RawDense { body: vec![1.0_f32;n * m] })))
            }
        }
    }

    pub fn new_from_vec(data: Vec<f32>, shape: Shape) -> Result<Self, ()> {
        match shape {
            Shape::D1(n) => {
                if data.len() != n {
                    return Err(());
                }
            },
            Shape::D2(r, c) => {
                if data.len() != r * c {
                    return Err(());
                }
            }
        }
        
        Ok(Self {
            name: String::new(),
            shape,
            storage: Arc::new(RwLock::new(Storage::Densef32(RawDense { body: data }))),
        })
    }

    pub fn storage(&self) -> RwLockReadGuard<'_, Storage> {
        self.storage.read().unwrap()
    }
    /* あぶないのでOptimiaer以外で使用禁止。実装もそっちにある
    pub fn storage_mut(&self) -> RwLockWriteGuard<'_, Storage> {
        self.storage.write().unwrap()
    }
    */

    // これはArc内部を書き換えるので注意！
    pub fn override_value(&self, new_value: Self) {
        let mut write = self.storage.write().unwrap();
        *write = new_value.storage().clone();
    }

    pub fn to_typed2d<const R: usize, const C: usize, T: Dtype>(&self) -> Result<Tensor2d<R, C, T>, String> {
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
            storage: Arc::new(RwLock::new(&*self.storage() + &*other.storage())),
        })
    }

    pub fn add_batch(&self) -> Self {
        // &*はRwLockReadGuard<'_, T>を&Tにしている
        let (body, col_num) = match &*self.storage() {
            Storage::Densef32(raw_dense) => {
                match self.shape {
                    Shape::D2(row_num, col_num) => {
                        let mut body = vec![0.0; col_num];
                        for r in 0..row_num {
                            for c in 0..col_num {
                                body[c] += raw_dense.body[r * col_num + c];
                            }
                        }
                        (body, col_num)
                    },
                    _ => {
                        LOGGER.error(format!("Tensor::add_batch() >> shape expection. shape is {}", self.shape.to_string()));
                        panic!("")
                    }
                }
            },
            _ => {
                LOGGER.error(format!("Tensor::add_batch() >> Storage type expection. self is {}", self.storage().info()));
                panic!("")
            },
        };
        Self {
            name: "add_batch".to_string(),
            shape: Shape::D1(col_num),
            storage: Storage::new_f32(body),
        }
    }

    // inplaceなのでmutにしてある（implaceはFnEdge内で使用禁止だが，これはOptimizerで使う）
    pub fn mul_scalar(&self, scalar: f32) -> Self {
        let new = match &*self.storage() {
            Storage::Densef32(raw_dense) => {
                let mut new = raw_dense.clone();
                new.mul_scalar(scalar);
                new
            },
            _ => {
                LOGGER.error(format!("Tensor::mul_scalar() >> Storage type expection. self is {}", self.storage().info()));
                panic!("")
            },
        };
        Self {
            name: self.name.clone(),
            shape: self.shape,
            storage: Storage::new_f32(new.body),
        }
    }
}  

impl std::ops::Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        let storage = &*self.storage() - &*rhs.storage();
        Self::Output {
            name: "tensor Sub".to_string(),
            shape: self.shape.clone(),
            storage: Arc::new(RwLock::new(storage)),
        }
    }
}
