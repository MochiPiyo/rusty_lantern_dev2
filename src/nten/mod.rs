


use std::fmt;

mod nten_matmul;
pub use nten_matmul::matmul;
mod nten;
pub use nten::Nten;
mod nten2d;
pub use nten2d::Nten2d;



#[derive(Eq, Hash, PartialEq, Clone, Copy)]
pub struct NtenID(pub i32);
pub(crate) fn get_new_nten_id(is_parameter: bool) -> NtenID {
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
    NtenID(new_id)
}
impl fmt::Display for NtenID {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub trait NtenTrait {
    
}