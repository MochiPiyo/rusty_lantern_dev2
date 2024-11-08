use std::fmt::{self, Debug};



// store bool as 1 bit not 1 Byte
#[derive(Clone, PartialEq)]
pub struct RawBool {
    pub len: usize,
    pub body: Vec<u8>,
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
    
    pub fn with_capacity(capacity: usize) -> Self {
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

    // Sets the value of a bit at a specific index.
    pub fn set_bit(&mut self, index: usize, value: bool) {
        let byte_pos = index / 8;
        let bit_pos = index % 8;

        // Ensure the byte_pos is within the bounds of the Vec.
        if byte_pos >= self.body.len() {
            panic!("RawDense set_bit() >> Index '{}' is out of bounds '{}'.", index, self.len);
        }

        if value {
            // Set the bit to 1.
            self.body[byte_pos] |= 1 << bit_pos;
        } else {
            // Set the bit to 0.
            self.body[byte_pos] &= !(1 << bit_pos);
        }
    }

    // Iterator method to enable iteration over the RawBool
    pub fn iter(&self) -> RawBoolIter {
        RawBoolIter {
            raw_bool: self,
            current_index: 0,
        }
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
// IndexAddは1 bitでやるのは難しい。変わりにset_bitがある


// Iterator for RawBool
pub struct RawBoolIter<'a> {
    raw_bool: &'a RawBool,
    current_index: usize,
}

impl<'a> Iterator for RawBoolIter<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if the current index is within the bounds of the RawBool
        if self.current_index < self.raw_bool.len {
            // Calculate byte and bit position
            let byte_pos = self.current_index / 8;
            let bit_pos = self.current_index % 8;

            // Access the byte and get the bit at the current index
            let byte = self.raw_bool.body[byte_pos];
            let bit_value = (byte >> bit_pos) & 1;

            // Increment the current index for the next iteration
            self.current_index += 1;

            // Return the boolean value of the bit
            Some(bit_value != 0)
        } else {
            // If the current index is out of bounds, return None to indicate the end of the iteration
            None
        }
    }
}