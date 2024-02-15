

mod storage;
pub use storage::*;
mod raw_tensor2d;
pub use raw_tensor2d::RawTensor2d;
mod untyped_raw_tensor;
pub use untyped_raw_tensor::RawTensor;

pub trait RawTensorTrait {
    fn mem_size(&self) -> usize;
}