use crate::{dtype::Shape, fn_edge::FnEdge, raw_tensor::RawTensor};

use super::TensorID;

// non typed version of Node
pub struct Tensor {
    pub id: TensorID,
    pub name: String,
    pub creator: Box<dyn FnEdge>,

    pub shape: Shape,
    pub val: Option<RawTensor>,
    pub grad: Option<RawTensor>,
}