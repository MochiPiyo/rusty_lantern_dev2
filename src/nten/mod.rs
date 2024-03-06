


use std::fmt;

mod nten_matmul;
pub use nten_matmul::matmul;
mod nten;
pub use nten::Nten;
mod nten2d;
pub use nten2d::Nten2d;


pub use crate::fn_edge::relu;

#[derive(Eq, Hash, PartialEq, Clone, Copy)]
pub struct NtenID(pub u32);
pub(crate) fn get_new_nten_id() -> NtenID {
    use std::sync::atomic;
    static COUNTER: atomic::AtomicU32 = atomic::AtomicU32::new(1);

    // get unique id
    // we separate parameter's id by make it minus
    /*
    Since parameters and gradients are managed internally using ID, 
    a model with branche swich, that have different generation order of Parameter Tensors
    occures problem at multi config execution.
    */
    let new_id = COUNTER.fetch_add(1, atomic::Ordering::Relaxed);
    NtenID(new_id)
}
impl fmt::Display for NtenID {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl fmt::Debug for NtenID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub trait NtenTrait {
    
}