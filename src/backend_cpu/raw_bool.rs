use std::fmt::{self, Debug};



// store bool as 1 bit not 1 Byte
#[derive(Clone, PartialEq)]
pub struct RawBool {
    len: usize,
    body: Vec<u8>,
}
impl Debug for RawBool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut bool_array = Vec::new();
        
        for i in 0..self.len {
            let byte_index = i / 8;
            let bit_index = i % 8;
            let mask = 1 << (7 - bit_index);
            let bit = (self.body[byte_index] & mask) != 0;
            bool_array.push(bit);
        }
        
        f.debug_list().entries(bool_array.iter()).finish()
    }
}
impl RawBool {
    pub fn new() -> Self {
        Self {
            len: 0,
            body: Vec::new(),
        }
    }
    
    pub fn new_with_capacity(capacity: usize) -> Self {
        // when capacity % 8 == 0, there is a extra 8 bit but it's not so big
        let bool_capacity = capacity / 8 + 1;
        Self {
            len: 0,
            body: Vec::with_capacity(bool_capacity),
        }
    }

    pub fn push(&mut self, val: bool) {
        let byte_pos = self.len / 8;
        let bit_pos = self.len % 8;

        if bit_pos == 0 {
            // If the current byte is full, add a new byte to the body Vec<u8>
            self.body.push(0);
        }

        // Update the bit at the specified position in the last byte
        if val {
            self.body[byte_pos] |= 1 << bit_pos;
        }

        // Increment the length
        self.len += 1;
    }

    pub fn new_from_vec(src: Vec<bool>) -> Self {
        let mut raw_bool = Self::new();
        for val in src {
            raw_bool.push(val);
        }
        raw_bool
    }
}

impl std::ops::Index<usize> for RawBool {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        // Calculate byte and bit position in the body Vec<u8>
        let byte_pos = index / 8;
        let bit_pos = index % 8;

        // Access the byte in the body Vec<u8> and get the bit at bit_pos
        let byte = &self.body[byte_pos];
        // Convert the bit at bit_pos to a boolean value
        let bit_value = (byte >> bit_pos) & 1;

        // Return the boolean value
        unsafe { &*(bit_value as *const u8 as *const bool) }
    }
}