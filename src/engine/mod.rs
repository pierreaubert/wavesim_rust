//! Engine module containing array abstractions and operations

pub mod array;
pub mod backend;
pub mod block;
pub mod operations;
pub mod sparse;

pub use array::{Complex64, WaveArray};
pub use operations::*;
