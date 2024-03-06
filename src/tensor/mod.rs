
mod matmul;
pub use matmul::matmul;
mod storage;
pub use storage::*;
mod tensor2d;
pub use tensor2d::Tensor2d;
mod tensor;
pub use tensor::Tensor;


pub trait TensorTrait {
    fn mem_size(&self) -> usize;
}