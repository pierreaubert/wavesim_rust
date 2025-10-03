//! Engine module containing array abstractions and operations

pub mod array;
pub mod block;
pub mod operations;
pub mod sparse;

pub use array::{Complex64, WaveArray};
pub use operations::*;
