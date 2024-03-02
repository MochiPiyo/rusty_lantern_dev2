

use std::fmt::Debug;

use crate::{backend_cpu::{RawBool, RawDense}, dtype::Shape, logger::LOGGER};

use std::ops::{Add, Sub, Div, Mul, Rem, AddAssign, SubAssign, DivAssign, MulAssign, RemAssign};

/*
-----------------Level 1, RawData---------------------------------
*/


// this is only for internal use in FrontDataType. for other use, use TypedRawData
// we collect all data type here. there are two parameter things, look like {structure type}{num type}, ex {Dense}{f32}
// for save in Vec<RawDataEnum>
#[derive(Clone, PartialEq)]
pub enum Storage {
    None, // if deleted

    DenseBool(RawBool),

    // candle have: u8, u32, i64, bf16, f16, f32, f64
    Densef32(RawDense<f32>), // we shold not use T here not to use generic parameter
    // Sparse32(..),
    // Gpu32(..),
}
// Noneで穴あきになると、epoch回す前にデフラグする必要がありそうだ。
impl Storage {
    // for count parameter num in GraphBuilder's parameter: Vec<RawData>
    pub fn parameter_num(&self) -> usize {
        todo!()
    }

    pub fn info(&self) -> &str {
        match self {
            Self::None => "RawData::None",
            Self::DenseBool(_) => "RawData::DenseBool",
            Self::Densef32(_) => "RawData::Densef32",
        }
    }

    pub fn matmul(lhs: &Self, lhs_shape: Shape, rhs: &Self, rhs_shape: Shape) -> Self {
        match (lhs, rhs) {
            (Storage::Densef32(lhs_dense), Storage::Densef32(rhs_dense)) => {
                let result_dense = RawDense::matmul(lhs_dense, lhs_shape, rhs_dense, rhs_shape);
                Storage::Densef32(result_dense)
            }
            // Handle other storage types and combinations
            _ => {
                LOGGER.error(format!("Storage::matmul() >> invalid pair. lhs: {}, rhs: {}", lhs.info(), rhs.info()));
                panic!("")
            },
        }
    }

    pub fn transpose(&self, shape: Shape) -> Self {
        match shape {
            Shape::D2(_, _) => {
                
            },
            _ => LOGGER.debug(format!("Storage::transpose() >> you execute transpose for shape: {}", shape.to_string()))
        }
        match self {
            Self::Densef32(raw) => {
                Self::Densef32(raw.transpose(shape))
            },
            Self::DenseBool(raw) => {
                todo!()
            }
            _ => {
                LOGGER.error(format!("Storage::transpose() >> Storage type expection. self is {}", self.info()));
                panic!("")
            },
        }
    }
}
impl Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "RawData::None"),
            Self::DenseBool(arg0) => f.debug_tuple("RawData::DenseBool").field(arg0).finish(),
            Self::Densef32(arg0) => f.debug_tuple("RawData::Densef32").field(arg0).finish(),
        }
    }
}


// Add, Sub, Div, Mul, Rem, AddAssign, SubAssigh, DivAssign, RemAssign
impl<'a> Add for &'a Storage {
    type Output = Storage;

    fn add(self, rhs: Self) -> Self::Output {
        use Storage::*;
        match (self, rhs) {
            (Densef32(lhs), Densef32(rhs)) => {
                return Densef32(lhs + rhs);
            },
            (lhs,  rhs) => panic!("Add not supported for lhs: '{}' and rhs: '{}'", lhs.info(), rhs.info()),
        }
    }
}

impl<'a> Sub for &'a Storage {
    type Output = Storage;

    fn sub(self, rhs: Self) -> Self::Output {
        use Storage::*;
        match (self, rhs) {
            (Densef32(lhs), Densef32(rhs)) => Densef32(lhs - rhs),
            (lhs, rhs) => panic!("Sub not supported for lhs: '{}' and rhs: '{}'", lhs.info(), rhs.info()),
        }
    }
}

impl<'a> Div for &'a Storage {
    type Output = Storage;

    fn div(self, rhs: Self) -> Self::Output {
        use Storage::*;
        match (self, rhs) {
            (Densef32(lhs), Densef32(rhs)) => Densef32(lhs / rhs),
            (lhs, rhs) => panic!("Div not supported for lhs: '{}' and rhs: '{}'", lhs.info(), rhs.info()),
        }
    }
}

impl<'a> Mul for &'a Storage {
    type Output = Storage;

    fn mul(self, rhs: Self) -> Self::Output {
        use Storage::*;
        match (self, rhs) {
            (Densef32(lhs), Densef32(rhs)) => Densef32(lhs * rhs),
            (lhs, rhs) => panic!("Mul not supported for lhs: '{}' and rhs: '{}'", lhs.info(), rhs.info()),
        }
    }
}

impl<'a> Rem for &'a Storage {
    type Output = Storage;

    fn rem(self, rhs: Self) -> Self::Output {
        use Storage::*;
        match (self, rhs) {
            (Densef32(lhs), Densef32(rhs)) => Densef32(lhs % rhs),
            (lhs, rhs) => panic!("Rem not supported for lhs: '{}' and rhs: '{}'", lhs.info(), rhs.info()),
        }
    }
}

impl AddAssign for Storage {
    fn add_assign(&mut self, rhs: Self) {
        match (self, rhs) {
            (Storage::Densef32(lhs), Storage::Densef32(rhs)) => {
                *lhs += rhs;
            },
            (lhs, rhs) => panic!("AddAssign not supported for lhs: '{}' and rhs: '{}'", lhs.info(), rhs.info()),
        }
    }
}

impl SubAssign for Storage {
    fn sub_assign(&mut self, rhs: Self) {
        match (self, rhs) {
            (Storage::Densef32(lhs), Storage::Densef32(rhs)) => {
                *lhs -= rhs;
            },
            (lhs, rhs) => panic!("SubAssign not supported for lhs: '{}' and rhs: '{}'", lhs.info(), rhs.info()),
        }
    }
}

impl DivAssign for Storage {
    fn div_assign(&mut self, rhs: Self) {
        match (self, rhs) {
            (Storage::Densef32(lhs), Storage::Densef32(rhs)) => {
                *lhs /= rhs;
            },
            (lhs, rhs) => panic!("DivAssign not supported for lhs: '{}' and rhs: '{}'", lhs.info(), rhs.info()),
        }
    }
}

impl MulAssign for Storage {
    fn mul_assign(&mut self, rhs: Self) {
        match (self, rhs) {
            (Storage::Densef32(lhs), Storage::Densef32(rhs)) => {
                *lhs *= rhs;
            },
            (lhs, rhs) => panic!("MulAssign not supported for lhs: '{}' and rhs: '{}'", lhs.info(), rhs.info()),
        }
    }
}

impl RemAssign for Storage {
    fn rem_assign(&mut self, rhs: Self) {
        match (self, rhs) {
            (Storage::Densef32(lhs), Storage::Densef32(rhs)) => {
                *lhs %= rhs;
            },
            (lhs, rhs) => panic!("RemAssign not supported for lhs: '{}' and rhs: '{}'", lhs.info(), rhs.info()),
        }
    }
}

