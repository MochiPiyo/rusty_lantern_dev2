use std::fmt::{format, Debug, Display, Formatter};


// トレイト'staticは内部に参照を含まないことを保証する
pub trait Dtype: Copy + Debug + 'static {
    fn default() -> Self;
    fn type_name() -> String;
}

impl Dtype for f32 {
    fn default() -> Self {
        0.0
    }

    fn type_name() -> String {
        "f32".to_string()
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum Shape {
    D1(usize),
    D2(usize, usize),
}
impl Shape {
    pub fn to_string(&self) -> String {
        match self {
            Self::D1(i) => format!("Shape::D1({})", i),
            Self::D2(i, j) => format!("Shape::D2({}, {})", i, j),
        }
    }
}
impl Display for Shape {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

// Sparse
pub struct Sf16;
pub type Sparsef16 = Sf16;
pub struct Sf32;
pub type Sparsef32 = Sf32;
pub struct Sf64;
pub type Sparsef64 = Sf64;

// GPU
pub struct Gf16;
pub struct Gf32;
pub struct Gf64;

// GPU Sparse
pub struct GSf16;
