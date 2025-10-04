//! Compute backend abstraction layer
//!
//! This module provides an abstraction over different computational backends
//! for FFT and vector operations. It allows seamless switching between:
//! - Apple Accelerate framework (macOS/iOS) - hardware-accelerated
//! - RustFFT (all platforms) - portable pure Rust implementation

use ndarray::Array3;
use num_complex::Complex;

// Backend implementations
#[cfg(all(feature = "accelerate", any(target_os = "macos", target_os = "ios")))]
mod accelerate;
mod rustfft;

#[cfg(all(feature = "accelerate", any(target_os = "macos", target_os = "ios")))]
pub use self::accelerate::AccelerateBackend;
pub use self::rustfft::RustFFTBackend;

/// Trait defining the compute backend interface
pub trait ComputeBackend: Send + Sync {
    /// Perform 3D forward FFT
    ///
    /// # Arguments
    /// * `input` - Input complex array
    /// * `output` - Output array (modified in place)
    fn fft_3d(&self, input: &Array3<Complex<f64>>, output: &mut Array3<Complex<f64>>);

    /// Perform 3D inverse FFT
    ///
    /// # Arguments
    /// * `input` - Input complex array
    /// * `output` - Output array (modified in place)
    fn ifft_3d(&self, input: &Array3<Complex<f64>>, output: &mut Array3<Complex<f64>>);

    /// Scale an array by a complex scalar and optionally add an offset
    /// output = scale * input + offset
    ///
    /// # Arguments
    /// * `scale` - Complex scaling factor
    /// * `input` - Input array
    /// * `offset` - Optional offset to add
    /// * `output` - Output array (modified in place)
    fn scale(
        &self,
        scale: Complex<f64>,
        input: &Array3<Complex<f64>>,
        offset: Option<Complex<f64>>,
        output: &mut Array3<Complex<f64>>,
    );

    /// Mix two arrays: output = alpha * a + beta * b
    ///
    /// # Arguments
    /// * `alpha` - Scaling factor for array a
    /// * `a` - First input array
    /// * `beta` - Scaling factor for array b
    /// * `b` - Second input array
    /// * `output` - Output array (modified in place)
    fn mix(
        &self,
        alpha: Complex<f64>,
        a: &Array3<Complex<f64>>,
        beta: Complex<f64>,
        b: &Array3<Complex<f64>>,
        output: &mut Array3<Complex<f64>>,
    );

    /// Linear interpolation: output = a + weight * (b - a)
    ///
    /// # Arguments
    /// * `a` - First input array
    /// * `b` - Second input array
    /// * `weight` - Weight array for interpolation
    /// * `output` - Output array (modified in place)
    fn lerp(
        &self,
        a: &Array3<Complex<f64>>,
        b: &Array3<Complex<f64>>,
        weight: &Array3<Complex<f64>>,
        output: &mut Array3<Complex<f64>>,
    );

    /// Return the name of the backend for debugging/logging
    fn name(&self) -> &'static str;
}

/// Get the default compute backend based on compile-time features and platform
pub fn default_backend() -> Box<dyn ComputeBackend> {
    #[cfg(all(feature = "accelerate", any(target_os = "macos", target_os = "ios")))]
    {
        Box::new(AccelerateBackend::new())
    }

    #[cfg(not(all(feature = "accelerate", any(target_os = "macos", target_os = "ios"))))]
    {
        Box::new(RustFFTBackend::new())
    }
}

/// Create a specific backend by name (useful for testing and benchmarking)
pub fn create_backend(name: &str) -> Option<Box<dyn ComputeBackend>> {
    match name {
        "rustfft" => Some(Box::new(RustFFTBackend::new())),
        #[cfg(all(feature = "accelerate", any(target_os = "macos", target_os = "ios")))]
        "accelerate" => Some(Box::new(AccelerateBackend::new())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_backend_creation() {
        let backend = default_backend();
        println!("Default backend: {}", backend.name());

        #[cfg(all(feature = "accelerate", any(target_os = "macos", target_os = "ios")))]
        assert_eq!(backend.name(), "accelerate");

        #[cfg(not(all(feature = "accelerate", any(target_os = "macos", target_os = "ios"))))]
        assert_eq!(backend.name(), "rustfft");
    }

    #[test]
    fn test_rustfft_backend_creation() {
        let backend = create_backend("rustfft");
        assert!(backend.is_some());
        assert_eq!(backend.unwrap().name(), "rustfft");
    }

    #[test]
    #[cfg(all(feature = "accelerate", any(target_os = "macos", target_os = "ios")))]
    fn test_accelerate_backend_creation() {
        let backend = create_backend("accelerate");
        assert!(backend.is_some());
        assert_eq!(backend.unwrap().name(), "accelerate");
    }
}
