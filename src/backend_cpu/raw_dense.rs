use rayon::prelude::*;
use std::{fmt::Debug, ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign}, sync::Mutex};
use lazy_static::lazy_static;

use crate::machine_config::{MACHINE_CONFIG, self};


// todo!
/*
加算とかの場合、
7950Xで1M要素くらいからトントン
1M = 1k * 1k

matmulとかは計算量がO(n^3)だからたぶん違う
*/

#[derive(Clone, Debug, PartialEq)]
pub struct RawDense<T> {
    // we don't need shape check because we invoke fn that impled this RawDense<T> only from Typed Tensor (front data type).

    // actual data
    // array[2][2] == [[a, b], 
    //                 [c, d]] 
    // -> vec![a, b, c, d]
    pub(crate) body: Vec<T>,
}

impl RawDense<f32>
{
    // use like self.template_op(other, "Raw Add", |a, b| a + b)
    /*
    operation: dyn Fn(T, T) -> T、実行時オーバーヘッドあり
    <F: Fn(T, T) -> T>(operation: F)、コンパイル時に型情報を解決、オーバーヘッドなし
     */
    #[inline(always)]
    fn template_op<F: Fn(f32, f32) -> f32 + std::marker::Sync>(&self, other: &Self, op_type: &str, operation: F) -> Self {
        if self.body.len() != other.body.len() {
            panic!("Error: failed to excute {op_type}. left body.len() is {} but ritht body.len() is {}", self.body.len(), other.body.len());
        }

        // excute
        let machine_config = MACHINE_CONFIG.lock().unwrap();
        let new_body = if self.body.len() > machine_config.multi_thread_threshold && machine_config.enable_multi_thread {
            // multi threaded
            self.body.par_iter().zip(&other.body).map(|(a, &b)| operation(*a, b)).collect()
        } else {
            // single threaded
            self.body.iter().zip(&other.body).map(|(a, b)| operation(*a, *b)).collect()
        };
        
        RawDense { body: new_body }
    }

    // operation for AddAssign, ...
    // + std::marker::Sync is needed at par_iter_mut() to send operation to other thread
    #[inline(always)]
    fn template_op_assign<'a, F: Fn(&mut f32, &f32) + std::marker::Sync>(&'a mut self, other: Self, op_type: &str, operation: F) {
        if self.body.len() != other.body.len() {
            panic!("Error: failed to excute {op_type}. left body.len() is {} but ritht body.len() is {}", self.body.len(), other.body.len());
        }

        let machine_config = MACHINE_CONFIG.lock().unwrap();
        if self.body.len() > machine_config.multi_thread_threshold && machine_config.enable_multi_thread {
            // multi threaded
            self.body.par_iter_mut().zip(other.body).for_each(|(a, b)| operation(a, &b));
        } else {
            // single threaded
            self.body.iter_mut().zip(&other.body).for_each(|(a, b)| operation(a, b));
        };
    }
}

// we implement non assign operations because RawData is mainly accessed through Rc which can't accept move ownership
impl<'a> Add for &'a RawDense<f32>
where f32: std::ops::Add<Output = f32> + Clone, {
    type Output = RawDense<f32>;
    fn add(self, other: Self) -> Self::Output {
        self.template_op(other, "Raw Add", |a, b| a + b)
    }
}

impl<'a> Sub for &'a RawDense<f32>
where f32: std::ops::Sub<Output = f32> + Clone, {
    type Output = RawDense<f32>;
    fn sub(self, other: Self) -> Self::Output {
        self.template_op(other, "Raw Sub", |a, b| a - b)
    }
}

impl<'a> Div for &'a RawDense<f32>
where f32: std::ops::Div<Output = f32> + Clone, {
    type Output = RawDense<f32>;
    fn div(self, other: Self) -> Self::Output {
        self.template_op(other, "Raw Div", |a, b| a / b)
    }
}

impl<'a> Mul for &'a RawDense<f32>
where f32: std::ops::Mul<Output = f32> + Clone, {
    type Output = RawDense<f32>;
    fn mul(self, other: Self) -> Self::Output {
        self.template_op(other, "Raw Mul", |a, b| a * b)
    }
}

impl<'a> Rem for &'a RawDense<f32>
where f32: std::ops::Rem<Output = f32> + Clone, {
    type Output = RawDense<f32>;
    fn rem(self, other: Self) -> Self::Output {
        self.template_op(other, "Raw Rem", |a, b| a % b)
    }
}



impl AddAssign for RawDense<f32> 
where f32: std::ops::AddAssign + Clone, {
    fn add_assign(&mut self, other: Self) {
        self.template_op_assign(other, "Raw AddAssign", |a, b| *a += *b);
    }
}

impl SubAssign for RawDense<f32>
where f32: std::ops::SubAssign + Clone, {
    fn sub_assign(&mut self, other: Self) {
        self.template_op_assign(other, "Raw SubAssign", |a, b| *a -= *b);
    }
}

impl DivAssign for RawDense<f32>
where f32: std::ops::DivAssign + Clone, {
    fn div_assign(&mut self, other: Self) {
        self.template_op_assign(other, "Raw DivAssign", |a, b| *a /= *b);
    }
}

impl MulAssign for RawDense<f32>
where f32: std::ops::MulAssign + Clone, {
    fn mul_assign(&mut self, other: Self) {
        self.template_op_assign(other, "Raw MulAssign", |a, b| *a *= *b);
    }
}

impl RemAssign for RawDense<f32>
where f32: std::ops::RemAssign + Clone, {
    fn rem_assign(&mut self, other: Self) {
        self.template_op_assign(other, "Raw RemAssign", |a, b| *a %= *b);
    }
}
