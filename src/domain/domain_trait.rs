//! Domain trait for wave equation solvers

use crate::engine::array::{Complex64, WaveArray};
use std::fmt::Debug;

/// Trait defining the interface for domain solvers
pub trait Domain: Debug + Send + Sync {
    /// Apply the medium operator (1 - V)
    fn medium(&self, x: &WaveArray<Complex64>, out: &mut WaveArray<Complex64>);

    /// Apply the propagator (L + 1)^{-1}
    fn propagator(&self, x: &WaveArray<Complex64>, out: &mut WaveArray<Complex64>);

    /// Apply the inverse propagator (L + 1)
    fn inverse_propagator(&self, x: &WaveArray<Complex64>, out: &mut WaveArray<Complex64>);

    /// Get the shape of the domain
    fn shape(&self) -> (usize, usize, usize);

    /// Get the scaling factor
    fn scale(&self) -> Complex64;

    /// Allocate a new array with the same shape as the domain
    fn allocate(&self, value: Complex64) -> WaveArray<Complex64> {
        WaveArray::from_scalar(self.shape(), value)
    }
}
