use std::{fmt::{format, Debug}, marker::PhantomData, sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard}};

use crate::{backend_cpu::{RawBool, RawDense}, dtype::{Dtype, Shape}, logger::LOGGER};

use super::{Tensor, Storage};

use colored::Colorize;
use rand::distributions::{Distribution, Uniform};
use rand_distr::Normal;

#[derive(Debug, Clone)]
pub struct Tensor2d<const R: usize, const C: usize, T> {
    pub name: String,
    pub storage: Arc<RwLock<Storage>>,
    pub _marker: PhantomData<T>,
}

impl<const R: usize, const C: usize, T: Dtype> Tensor2d<R, C, T> {
    pub fn new_zeros() -> Self {
        if T::type_name() == "f32".to_string() {
            let mut body = Vec::new();
            for i in 0..R {
                for j in 0..C {
                    body.push(0.0);
                }
            }
            let raw_dense = RawDense { body };
            Self {
                name: "no_name".to_string(),
                storage: Arc::new(RwLock::new(Storage::Densef32(raw_dense))),
                _marker: PhantomData,
            }
        } else {
            LOGGER.error(format!("{}::{}() >> not suppoerted T", Self::type_name().green(), "new_zeros".yellow()));
            panic!();
        }
    }
    pub fn new_ones() -> Self {
        if T::type_name() == "f32".to_string() {
            let mut body = Vec::new();
            for i in 0..R {
                for j in 0..C {
                    body.push(1.0);
                }
            }
            let raw_dense = RawDense { body };
            Self {
                name: "no_name".to_string(),
                storage: Arc::new(RwLock::new(Storage::Densef32(raw_dense))),
                _marker: PhantomData,
            }
        } else {
            LOGGER.error(format!("{}::{}() >> not suppoerted T", Self::type_name().green(), "new_zeros".yellow()));
            panic!();
        }
        
    }

    pub fn type_name() -> String {
        format!("Tensor2d<{}, {}, {}>", R, C, T::type_name())
    }

    pub fn storage(&self) -> RwLockReadGuard<'_, Storage> {
        self.storage.read().unwrap()
    }
    /* あぶないのでOptimiaer以外で使用禁止
    pub fn storage_mut(&self) -> RwLockWriteGuard<'_, Storage> {
        self.storage.write().unwrap()
    }*/

    // これはArc内部を書き換えるので注意！
    pub fn override_value(&self, new_value: Self) {
        let mut write = self.storage.write().unwrap();
        *write = new_value.storage().clone();
    }
    
    pub fn to_untyped(&self) -> Tensor {
        Tensor {
            name: self.name.clone(),
            shape: Shape::D2(R, C),
            storage: self.storage.clone(),
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
            storage: Arc::new(RwLock::new(&*self.storage() + &*other.storage())),
            _marker: PhantomData,
        }
    }

    pub fn add_broadcast(&self, bias: &Tensor2d<1, C, T>) -> Self {
        // &*はRwLockReadGuard<'_, T>を&Tにしている
        match (&*self.storage(), &*bias.storage()) {
            (Storage::Densef32(raw), Storage::Densef32(raw_bias)) => {
                let mut new = raw.clone();
                new.add_broadcast(raw_bias, Shape::D2(R, C));
                Self {
                    name: "add_broadcast".to_string(),
                    storage: Storage::new_f32(new.body),
                    _marker: PhantomData,
                }
            },
            (_, _) => {
                LOGGER.error(format!("Tensor2d<{}, {}, {}> >> unsupported Storage type. lhs name: '{}', rhs name: '{}'", R, C, T::type_name(), self.name, bias.name));
                panic!()
            },
        }
    }

    pub fn sum_batch(&self) -> Tensor2d<1, C, T> {
        // &*はRwLockReadGuard<'_, T>を&Tにしている
        match &*self.storage() {
            Storage::Densef32(raw) => {
                let new = raw.sum_batch(Shape::D2(R, C));
                Tensor2d::<1, C, T> {
                    name: "sum_batch".to_string(),
                    storage: Storage::new_f32(new.body),
                    _marker: PhantomData,
                }
            },
            _ => {
                LOGGER.error(format!("{}::{}() >> Storage type expection. {} is not supported",
                    Self::type_name().green(), "sum_batch".yellow(),
                    self.storage().info()));
                panic!("")
            }
        }
    }

    /* 関数を渡すとGPUで対応できないのでこれは却下
    // TにするとTで場合分けしてrawにわたすことになって，型の変換がunsafeになる
    // とりあえずのところ，f32
    pub fn replace_where_f32(self, selector: fn(f32) -> bool, replace_to: f32) -> Self {
        if let Storage::Densef32(raw) = &mut *self.storage_mut() {
            raw.replace_where(selector, replace_to)
        } else {
            LOGGER.error(format!("Tensor2d<{}, {}, f32>::relu() >> Storage type expection. {} is not supported", R, C, self.storage().info()));
                panic!("")
        }
        self
    }
    */

    // larger element is true
    pub fn select_larger_than(&self, condition: T) -> Tensor2d<R, C, bool> {
        // &*はRwLockReadGuard<'_, T>を&Tにしている
        match &*self.storage() {
            Storage::Densef32(raw) => {
                // convert T to f32
                if let Ok(x) = condition.to_f32() {
                    let raw_bool = raw.select_larger_than(x);
                    Tensor2d::<R, C, bool> {
                        name: "select_larger_than".to_string(),
                        storage: Storage::new_bools(raw_bool.body, raw_bool.len),
                        _marker: PhantomData,
                    }
                } else {
                    LOGGER.error(format!("Tensor2d<{}, {}, {}>::select_larger_than() >> unexpected error. failed to convert condition: T to f32", R, C, T::type_name()));
                    panic!("")
                }
            },
            _ => {
                LOGGER.error(format!("{}::{}() >> Storage type expection. {} is not supported",
                    Self::type_name().green(), "select_larger_than".yellow(),
                    self.storage().info()));
                panic!("")
            }
        }
        
    }

    pub fn select_smaller_than(&self, condition: T) -> Tensor2d<R, C, bool> {
        // &*はRwLockReadGuard<'_, T>を&Tにしている
        match &*self.storage() {
            Storage::Densef32(raw) => {
                // convert T to f32
                if let Ok(x) = condition.to_f32() {
                    let raw_bool = raw.select_smaller_than(x);
                    Tensor2d::<R, C, bool> {
                        name: "select_smaller_than".to_string(),
                        storage: Storage::new_bools(raw_bool.body, raw_bool.len),
                        _marker: PhantomData,
                    }
                } else {
                    LOGGER.error(format!("Tensor2d<{}, {}, {}>::select_smaller_than() >> unexpected error. failed to convert condition: T to f32", R, C, T::type_name()));
                    panic!("")
                }
            },
            _ => {
                LOGGER.error(format!("{}::{}() >> Storage type expection. {} is not supported",
                    Self::type_name().green(), "select_smaller_than".yellow(),
                    self.storage().info()));
                panic!("")
            }
        }
    }

    // replace element where mask is true to valeue of "to"
    pub fn replace_scalar_where(&self, mask: &Tensor2d<R, C, bool>, to: T) -> Self {
        // &*はRwLockReadGuard<'_, T>を&Tにしている
        match &*self.storage() {
            Storage::Densef32(raw) => {
                let mut new = raw.clone();
                // ここはわけないとtemporary valueがdropする
                let bool_storage = mask.storage();
                let raw_mask = if let Storage::DenseBool(raw) = &*bool_storage {
                    raw
                } else {
                    LOGGER.error(format!("{}::{}() >> mask tensor not have bool but {}",
                            Self::type_name().green(), "replace_scalar_where".yellow(),
                            mask.storage().info()));
                    panic!("")
                };
                new.replace_where_to_scalar(raw_mask, to.to_f32().unwrap());
                Self {
                    name: "replace_scalar_where".to_string(),
                    storage: Storage::new_f32(new.body),
                    _marker: PhantomData,
                }
            },
            _ => {
                LOGGER.error(format!("{}::{}() >> Storage type expection. {} is not supported",
                    Self::type_name().green(), "replace_scalar_where".yellow(),
                    self.storage().info()));
                panic!("")
            }
        }
    }

    // replace element where mask is true to value of to: Self
    pub fn replace_where(&self, mask: &Tensor2d<R, C, bool>, to: Self) -> Self {
        todo!()
    }

    pub fn transpose(&self) -> Tensor2d<C, R, T> {
        // &*はRwLockReadGuard<'_, T>を&Tにしている
        let new = match &*self.storage() {
            Storage::Densef32(raw) => {
                let mut new = raw.clone();
                new.transpose(Shape::D2(R, C));
                new
            },
            Storage::DenseBool(raw) => {
                todo!()
            }
            Storage::None => {
                LOGGER.error(format!("Tensor2d<{}, {}, {}>::transpose() >> Storage type expection. {} is not supported", R, C, T::type_name(), self.storage().info()));
                panic!("")
            },
        };

        // <R, C, T> to <C, R, T>
        Tensor2d::<C, R, T> {
            name: format!("transposed from '{}'", self.name),
            storage: Storage::new_f32(new.body),
            _marker: PhantomData,
        }
    }

    pub fn top_index_per_batch(&self) -> Vec<usize> {
        match &*self.storage() {
            Storage::Densef32(raw) => {
                let mut indices = Vec::new();
                for data in raw.body.chunks(C) {
                    let mut largest = f32::MIN;
                    let mut index: usize = 0;
                    for (i, val) in data.iter().enumerate() {
                        if *val > largest {
                            largest = *val;
                            index = i;
                        }
                    }
                    indices.push(index);
                }
                
                indices
            },
            _ => {
                LOGGER.error(format!("{}::{}() >> Storage type expection. {} is not supported",
                    Self::type_name().green(), "top_index_per_batch".yellow(),
                    self.storage().info()));
                panic!("")
            }
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
            storage: Arc::new(RwLock::new(Storage::Densef32(raw_dense))),
            _marker: PhantomData,
        }
    }

    pub fn new_from_vec(data: Vec<f32>) -> Result<Self, ()> {
        if data.len() != R * C {
            return Err(());
        }
        Ok(Self {
            name: String::new(),
            storage: Arc::new(RwLock::new(Storage::Densef32(RawDense { body: data }))),
            _marker: PhantomData,
        })
    }

    pub fn new_uniform(low: f32, high: f32) -> Self {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(low, high);
        let random: Vec<f32> = (0..R * C).map(|_| uniform.sample(&mut rng)).collect();
        
        Self {
            name: "created by new_uniform()".to_string(),
            storage: Arc::new(RwLock::new(Storage::Densef32(RawDense { body: random }))),
            _marker: PhantomData,
        }
    }
    pub fn new_normal(mean: f64, std_dev: f64) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(mean as f64, std_dev as f64).unwrap();
        let random: Vec<f32> = (0..R*C).map(|_| normal.sample(&mut rng) as f32).collect();
        
        Self {
            name: "created by new_normal()".to_string(),
            storage: Arc::new(RwLock::new(Storage::Densef32(RawDense { body: random }))),
            _marker: PhantomData,
        }
    }
    pub fn new_init_he() -> Self {
        let mean = 0.0;
        let std_dev = (2.0 / R as f64).sqrt();

        let mut rng = rand::thread_rng();
        let normal = Normal::new(mean as f64, std_dev as f64).unwrap();
        let random: Vec<f32> = (0..R*C).map(|_| normal.sample(&mut rng) as f32).collect();
        
        Self {
            name: "created by new_init_he()".to_string(),
            storage: Arc::new(RwLock::new(Storage::Densef32(RawDense { body: random }))),
            _marker: PhantomData,
        }
    }

    

    
}


impl<const R: usize, const C: usize> Tensor2d<R, C, bool> {
    pub fn new_tures() -> Self {
        let len = (R * C) / 8 + 1;
        let bools: Vec<u8> = vec![0b1111_1111; len];
        Self {
            name: "bool new_trues".to_string(),
            storage: Storage::new_bools(bools, len),
            _marker: PhantomData,
        }
    }
    pub fn new_falses() -> Self {
        let len = (R * C) / 8 + 1;
        let bools: Vec<u8> = vec![0b0000_0000; len];
        Self {
            name: "bool new_trues".to_string(),
            storage: Storage::new_bools(bools, len),
            _marker: PhantomData,
        }
    }

    pub fn count_true(&self) -> usize {
        match &*self.storage() {
            Storage::DenseBool(raw) => {
                let mut counter = 0;
                for b in raw.iter() {
                    if b == true {
                        counter += 1;
                    }
                }
                counter
            },
            _ => {
                LOGGER.error(format!("{}::{}() >> Storage type expection. {} is not supported",
                    Self::type_name().green(), "count_true".yellow(),
                    self.storage().info()));
                panic!("")
            }
        }
    }
}