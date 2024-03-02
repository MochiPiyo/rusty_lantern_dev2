use crate::{dtype::Shape, fn_edge::FnEdge, tensor::{self, Tensor}};

use super::NtenID;

// non typed version of Node
#[derive(Clone)]
pub struct Nten {
    pub id: NtenID,
    pub name: String,
    pub creator: Box<dyn FnEdge>,

    pub shape: Shape,
    pub val: Option<Tensor>,
    pub grad: Option<Tensor>,
}
impl Nten {
    pub fn name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

}