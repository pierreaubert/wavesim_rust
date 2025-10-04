//! Array operations for wave simulations
//!
//! This module provides high-level array operations that automatically dispatch
//! to the appropriate backend (Accelerate on Apple platforms, RustFFT elsewhere).

use crate::engine::array::{Complex64, WaveArray};
use crate::engine::backend::{default_backend, ComputeBackend};
use ndarray::{Array3, Zip};
use num_traits::Zero;
use once_cell::sync::Lazy;
use std::sync::Arc;

// Global backend instance (thread-safe, initialized once)
static BACKEND: Lazy<Arc<Box<dyn ComputeBackend>>> = Lazy::new(|| Arc::new(default_backend()));

/// Perform element-wise multiplication
pub fn multiply(a: &WaveArray<Complex64>, b: &WaveArray<Complex64>) -> WaveArray<Complex64> {
    WaveArray {
        data: &a.data * &b.data,
    }
}

/// Perform element-wise division
pub fn divide(a: &WaveArray<Complex64>, b: &WaveArray<Complex64>) -> WaveArray<Complex64> {
    WaveArray {
        data: &a.data / &b.data,
    }
}

/// Scale an array by a complex scalar and add an offset
/// out = scale * input + offset
///
/// Uses the optimized backend for better performance on supported platforms.
pub fn scale(
    scale: Complex64,
    input: &WaveArray<Complex64>,
    offset: Option<Complex64>,
    out: &mut WaveArray<Complex64>,
) {
    BACKEND.scale(scale, &input.data, offset, &mut out.data);
}

/// Mix two arrays: out = alpha * a + beta * b
///
/// Uses the optimized backend for better performance on supported platforms.
pub fn mix(
    alpha: Complex64,
    a: &WaveArray<Complex64>,
    beta: Complex64,
    b: &WaveArray<Complex64>,
    out: &mut WaveArray<Complex64>,
) {
    BACKEND.mix(alpha, &a.data, beta, &b.data, &mut out.data);
}

/// Linear interpolation: out = a + weight * (b - a)
///
/// Uses the optimized backend for better performance on supported platforms.
pub fn lerp(
    a: &WaveArray<Complex64>,
    b: &WaveArray<Complex64>,
    weight: &WaveArray<Complex64>,
    out: &mut WaveArray<Complex64>,
) {
    BACKEND.lerp(&a.data, &b.data, &weight.data, &mut out.data);
}

/// Copy data from one array to another
pub fn copy(source: &WaveArray<Complex64>, dest: &mut WaveArray<Complex64>) {
    dest.data.assign(&source.data);
}

/// Perform 3D FFT
///
/// This function automatically uses the best available backend:
/// - Apple Accelerate framework on macOS/iOS (when compiled with --features accelerate)
/// - RustFFT on all other platforms or when Accelerate is not enabled
pub fn fft_3d(input: &WaveArray<Complex64>, output: &mut WaveArray<Complex64>) {
    BACKEND.fft_3d(&input.data, &mut output.data);
}

/// Perform 3D inverse FFT
///
/// This function automatically uses the best available backend:
/// - Apple Accelerate framework on macOS/iOS (when compiled with --features accelerate)
/// - RustFFT on all other platforms or when Accelerate is not enabled
pub fn ifft_3d(input: &WaveArray<Complex64>, output: &mut WaveArray<Complex64>) {
    BACKEND.ifft_3d(&input.data, &mut output.data);
}

/// Matrix multiplication along specified axis
pub fn matmul(
    matrix: &Array3<Complex64>,
    x: &WaveArray<Complex64>,
    axis: usize,
    out: &mut WaveArray<Complex64>,
) {
    // This is a simplified version - in production we'd use BLAS
    let shape = x.shape();

    match axis {
        0 => {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    for i_out in 0..matrix.shape()[0] {
                        let mut sum = Complex64::zero();
                        for i_in in 0..matrix.shape()[1] {
                            sum += matrix[[i_out, i_in, 0]] * x.data[[i_in, j, k]];
                        }
                        out.data[[i_out, j, k]] = sum;
                    }
                }
            }
        }
        1 => {
            for i in 0..shape[0] {
                for k in 0..shape[2] {
                    for j_out in 0..matrix.shape()[0] {
                        let mut sum = Complex64::zero();
                        for j_in in 0..matrix.shape()[1] {
                            sum += matrix[[j_out, j_in, 0]] * x.data[[i, j_in, k]];
                        }
                        out.data[[i, j_out, k]] = sum;
                    }
                }
            }
        }
        2 => {
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    for k_out in 0..matrix.shape()[0] {
                        let mut sum = Complex64::zero();
                        for k_in in 0..matrix.shape()[1] {
                            sum += matrix[[k_out, k_in, 0]] * x.data[[i, j, k_in]];
                        }
                        out.data[[i, j, k_out]] = sum;
                    }
                }
            }
        }
        _ => panic!("Invalid axis"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_scale_operation() {
        let input = WaveArray::from_scalar((2, 2, 2), Complex64::new(1.0, 0.0));
        let mut output = WaveArray::zeros((2, 2, 2));

        scale(
            Complex64::new(2.0, 0.0),
            &input,
            Some(Complex64::new(1.0, 0.0)),
            &mut output,
        );

        assert_eq!(output.data[[0, 0, 0]], Complex64::new(3.0, 0.0));
    }

    #[test]
    fn test_mix_operation() {
        let a = WaveArray::from_scalar((2, 2, 2), Complex64::new(1.0, 0.0));
        let b = WaveArray::from_scalar((2, 2, 2), Complex64::new(2.0, 0.0));
        let mut output = WaveArray::zeros((2, 2, 2));

        mix(
            Complex64::new(2.0, 0.0),
            &a,
            Complex64::new(3.0, 0.0),
            &b,
            &mut output,
        );

        assert_eq!(output.data[[0, 0, 0]], Complex64::new(8.0, 0.0));
    }
}
