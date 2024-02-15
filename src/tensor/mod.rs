


use std::fmt;

mod tensor2d;
pub use tensor2d::Tensor2d;

mod untyped_tensor;
pub use untyped_tensor::Tensor;

#[derive(Eq, Hash, PartialEq, Clone, Copy)]
pub struct TensorID(pub i32);
fn get_new_tensor_id(is_parameter: bool) -> TensorID {
    use std::sync::atomic;
    static COUNTER: atomic::AtomicU32 = atomic::AtomicU32::new(1);
    static PARAMETER_ID_COUNTER: atomic::AtomicU32 = atomic::AtomicU32::new(1);

    // get unique id
    // we separate parameter's id by make it minus
    /*
    Since parameters and gradients are managed internally using ID, 
    a model with branche swich, that have different generation order of Parameter Tensors
    occures problem at multi config execution.
    */
    let new_id = if is_parameter {
        let u32 = PARAMETER_ID_COUNTER.fetch_add(1, atomic::Ordering::Relaxed);
        // important minus !
        -(u32 as i32)
    } else {
        let u32 = COUNTER.fetch_add(1, atomic::Ordering::Relaxed);
        u32 as i32
    };
    TensorID(new_id)
}
impl fmt::Display for TensorID {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub trait TensorTrait {
    
}