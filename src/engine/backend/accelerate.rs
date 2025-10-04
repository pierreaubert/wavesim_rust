//! Accelerate framework backend implementation
//!
//! Hardware-accelerated implementation using Apple's Accelerate framework.
//! This backend is only available on macOS and iOS.

#![cfg(any(target_os = "macos", target_os = "ios"))]

// Link to the Accelerate framework
#[link(name = "Accelerate", kind = "framework")]
extern "C" {}

use super::{ComputeBackend, RustFFTBackend};
use ndarray::Array3;
use num_complex::Complex;
use once_cell::sync::Lazy;
use std::collections::HashSet;
use std::os::raw::{c_int, c_void};
use std::sync::Mutex;

// FFI declarations for Accelerate framework vDSP functions
// These are complex double-precision functions
#[repr(C)]
struct DSPDoubleSplitComplex {
    realp: *mut f64,
    imagp: *mut f64,
}

// Using accelerate-src crate which provides the actual linkage
// We declare the functions we need from vDSP
extern "C" {
    // Complex vector operations
    fn vDSP_zvaddD(
        __A: *const DSPDoubleSplitComplex,
        __IA: c_int,
        __B: *const DSPDoubleSplitComplex,
        __IB: c_int,
        __C: *const DSPDoubleSplitComplex,
        __IC: c_int,
        __N: usize,
    );

    fn vDSP_zvmulD(
        __A: *const DSPDoubleSplitComplex,
        __IA: c_int,
        __B: *const DSPDoubleSplitComplex,
        __IB: c_int,
        __C: *const DSPDoubleSplitComplex,
        __IC: c_int,
        __N: usize,
        __Conjugate: c_int,
    );

    fn vDSP_zvsmaD(
        __A: *const DSPDoubleSplitComplex,
        __IA: c_int,
        __B: *const f64,
        __C: *const DSPDoubleSplitComplex,
        __IC: c_int,
        __N: usize,
    );

    fn vDSP_zvmaD(
        __A: *const DSPDoubleSplitComplex,
        __IA: c_int,
        __B: *const DSPDoubleSplitComplex,
        __IB: c_int,
        __C: *const DSPDoubleSplitComplex,
        __IC: c_int,
        __D: *const DSPDoubleSplitComplex,
        __ID: c_int,
        __N: usize,
    );

    // FFT setup and execution
    fn vDSP_create_fftsetupD(__Log2n: usize, __Radix: c_int) -> *mut c_void;

    fn vDSP_destroy_fftsetupD(__setup: *mut c_void);

    fn vDSP_fft_zripD(
        __Setup: *mut c_void,
        __C: *const DSPDoubleSplitComplex,
        __IC: c_int,
        __Log2N: usize,
        __Direction: c_int,
    );

    fn vDSP_fft_zropD(
        __Setup: *mut c_void,
        __A: *const DSPDoubleSplitComplex,
        __IA: c_int,
        __C: *const DSPDoubleSplitComplex,
        __IC: c_int,
        __Log2N: usize,
        __Direction: c_int,
    );
}

const FFT_FORWARD: c_int = 1;
const FFT_INVERSE: c_int = -1;

// Global cache to track which shapes have already been warned about
// This prevents spamming the same warning on every iteration
static WARNED_SHAPES: Lazy<Mutex<HashSet<(usize, usize, usize)>>> =
    Lazy::new(|| Mutex::new(HashSet::new()));

/// Accelerate-based compute backend with automatic fallback to RustFFT
pub struct AccelerateBackend {
    /// Fallback backend for non-power-of-2 sizes
    fallback: RustFFTBackend,
}

impl AccelerateBackend {
    /// Create a new Accelerate backend with RustFFT fallback
    pub fn new() -> Self {
        Self {
            fallback: RustFFTBackend::new(),
        }
    }

    /// Check if all dimensions are powers of 2
    fn is_power_of_2_shape(shape: &[usize]) -> bool {
        shape.iter().all(|&dim| dim > 0 && (dim & (dim - 1)) == 0)
    }

    /// Convert interleaved complex array to split complex format
    fn to_split_complex(data: &[Complex<f64>]) -> (Vec<f64>, Vec<f64>) {
        let real: Vec<f64> = data.iter().map(|c| c.re).collect();
        let imag: Vec<f64> = data.iter().map(|c| c.im).collect();
        (real, imag)
    }

    /// Convert split complex format back to interleaved
    fn from_split_complex(real: &[f64], imag: &[f64]) -> Vec<Complex<f64>> {
        real.iter()
            .zip(imag.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect()
    }

    /// Perform 1D FFT along a specific axis using vDSP
    /// Assumes power-of-2 size check has already been done
    fn fft_1d_axis(data: &mut Array3<Complex<f64>>, axis: usize, direction: c_int) {
        let shape = data.shape().to_vec(); // Copy shape to avoid borrow issues
        let n = shape[axis];

        let log2n = n.trailing_zeros() as usize;

        unsafe {
            let setup = vDSP_create_fftsetupD(log2n, 2); // Radix-2
            if setup.is_null() {
                panic!("Failed to create FFT setup");
            }

            match axis {
                0 => {
                    for j in 0..shape[1] {
                        for k in 0..shape[2] {
                            let mut slice: Vec<Complex<f64>> =
                                (0..n).map(|i| data[[i, j, k]]).collect();

                            let (mut real, mut imag) = Self::to_split_complex(&slice);

                            let split = DSPDoubleSplitComplex {
                                realp: real.as_mut_ptr(),
                                imagp: imag.as_mut_ptr(),
                            };

                            vDSP_fft_zropD(setup, &split, 1, &split, 1, log2n, direction);

                            let result = Self::from_split_complex(&real, &imag);
                            for (i, val) in result.iter().enumerate() {
                                data[[i, j, k]] = *val;
                            }
                        }
                    }
                }
                1 => {
                    for i in 0..shape[0] {
                        for k in 0..shape[2] {
                            let mut slice: Vec<Complex<f64>> =
                                (0..n).map(|j| data[[i, j, k]]).collect();

                            let (mut real, mut imag) = Self::to_split_complex(&slice);

                            let split = DSPDoubleSplitComplex {
                                realp: real.as_mut_ptr(),
                                imagp: imag.as_mut_ptr(),
                            };

                            vDSP_fft_zropD(setup, &split, 1, &split, 1, log2n, direction);

                            let result = Self::from_split_complex(&real, &imag);
                            for (j, val) in result.iter().enumerate() {
                                data[[i, j, k]] = *val;
                            }
                        }
                    }
                }
                2 => {
                    for i in 0..shape[0] {
                        for j in 0..shape[1] {
                            let mut slice: Vec<Complex<f64>> =
                                (0..n).map(|k| data[[i, j, k]]).collect();

                            let (mut real, mut imag) = Self::to_split_complex(&slice);

                            let split = DSPDoubleSplitComplex {
                                realp: real.as_mut_ptr(),
                                imagp: imag.as_mut_ptr(),
                            };

                            vDSP_fft_zropD(setup, &split, 1, &split, 1, log2n, direction);

                            let result = Self::from_split_complex(&real, &imag);
                            for (k, val) in result.iter().enumerate() {
                                data[[i, j, k]] = *val;
                            }
                        }
                    }
                }
                _ => unreachable!(),
            }

            vDSP_destroy_fftsetupD(setup);
        }
    }
}

impl Default for AccelerateBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for AccelerateBackend {
    fn fft_3d(&self, input: &Array3<Complex<f64>>, output: &mut Array3<Complex<f64>>) {
        // Check if dimensions are power-of-2; if not, use RustFFT fallback
        if !Self::is_power_of_2_shape(input.shape()) {
            // Warn only once per unique shape to avoid spam
            let shape = input.shape();
            let shape_tuple = (shape[0], shape[1], shape[2]);

            if let Ok(mut warned) = WARNED_SHAPES.lock() {
                if warned.insert(shape_tuple) {
                    eprintln!("\n⚠️  Performance Warning: Accelerate Backend Fallback");
                    eprintln!(
                        "   Shape {:?} is not power-of-2. Using RustFFT (5-10x slower).",
                        shape_tuple
                    );
                    eprintln!(
                        "   💡 Tip: Use wavesim::utilities::domain_sizing::optimal_domain_shape()"
                    );
                    eprintln!("        for optimal performance on Apple platforms.\n");
                }
            }
            return self.fallback.fft_3d(input, output);
        }

        output.assign(input);

        // Perform FFT along each axis
        Self::fft_1d_axis(output, 0, FFT_FORWARD);
        Self::fft_1d_axis(output, 1, FFT_FORWARD);
        Self::fft_1d_axis(output, 2, FFT_FORWARD);
    }

    fn ifft_3d(&self, input: &Array3<Complex<f64>>, output: &mut Array3<Complex<f64>>) {
        // Check if dimensions are power-of-2; if not, use RustFFT fallback
        if !Self::is_power_of_2_shape(input.shape()) {
            // Note: We only warn on the first FFT call (forward), not on every inverse FFT
            // to avoid spam during iteration loops
            return self.fallback.ifft_3d(input, output);
        }

        output.assign(input);

        // Perform inverse FFT along each axis
        Self::fft_1d_axis(output, 0, FFT_INVERSE);
        Self::fft_1d_axis(output, 1, FFT_INVERSE);
        Self::fft_1d_axis(output, 2, FFT_INVERSE);

        // Apply normalization
        // vDSP applies a scale of 2 for each transform, so we need to divide by 2^(3 dimensions)
        // Plus the standard FFT normalization of 1/N
        let shape = output.shape();
        let n_total = (shape[0] * shape[1] * shape[2]) as f64;
        let normalization = 1.0 / (n_total * 8.0); // 8 = 2^3 for the vDSP scaling
        for elem in output.iter_mut() {
            *elem *= normalization;
        }
    }

    fn scale(
        &self,
        scale: Complex<f64>,
        input: &Array3<Complex<f64>>,
        offset: Option<Complex<f64>>,
        output: &mut Array3<Complex<f64>>,
    ) {
        let n = input.len();
        let (input_real, input_imag) = Self::to_split_complex(input.as_slice().unwrap());
        let mut out_real = vec![0.0; n];
        let mut out_imag = vec![0.0; n];

        // Complex multiplication: output = scale * input
        // We need to do this manually since vDSP doesn't have a direct complex scalar multiply

        // For complex multiplication (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        // where scale = (a + bi) and input elements = (c + di)

        let a = scale.re;
        let b = scale.im;

        for i in 0..n {
            let c = input_real[i];
            let d = input_imag[i];
            out_real[i] = a * c - b * d;
            out_imag[i] = a * d + b * c;
        }

        if let Some(off) = offset {
            // Add offset
            for i in 0..n {
                out_real[i] += off.re;
                out_imag[i] += off.im;
            }
        }

        let result = Self::from_split_complex(&out_real, &out_imag);
        for (i, val) in result.iter().enumerate() {
            output.as_slice_mut().unwrap()[i] = *val;
        }
    }

    fn mix(
        &self,
        alpha: Complex<f64>,
        a: &Array3<Complex<f64>>,
        beta: Complex<f64>,
        b: &Array3<Complex<f64>>,
        output: &mut Array3<Complex<f64>>,
    ) {
        let n = a.len();

        // Convert to split complex
        let (a_real, a_imag) = Self::to_split_complex(a.as_slice().unwrap());
        let (b_real, b_imag) = Self::to_split_complex(b.as_slice().unwrap());

        let mut temp_a_real = vec![0.0; n];
        let mut temp_a_imag = vec![0.0; n];
        let mut temp_b_real = vec![0.0; n];
        let mut temp_b_imag = vec![0.0; n];
        let mut out_real = vec![0.0; n];
        let mut out_imag = vec![0.0; n];

        // Compute alpha * a
        for i in 0..n {
            let c = a_real[i];
            let d = a_imag[i];
            temp_a_real[i] = alpha.re * c - alpha.im * d;
            temp_a_imag[i] = alpha.re * d + alpha.im * c;
        }

        // Compute beta * b
        for i in 0..n {
            let c = b_real[i];
            let d = b_imag[i];
            temp_b_real[i] = beta.re * c - beta.im * d;
            temp_b_imag[i] = beta.re * d + beta.im * c;
        }

        // Add them together
        for i in 0..n {
            out_real[i] = temp_a_real[i] + temp_b_real[i];
            out_imag[i] = temp_a_imag[i] + temp_b_imag[i];
        }

        let result = Self::from_split_complex(&out_real, &out_imag);
        for (i, val) in result.iter().enumerate() {
            output.as_slice_mut().unwrap()[i] = *val;
        }
    }

    fn lerp(
        &self,
        a: &Array3<Complex<f64>>,
        b: &Array3<Complex<f64>>,
        weight: &Array3<Complex<f64>>,
        output: &mut Array3<Complex<f64>>,
    ) {
        let n = a.len();

        // output = a + weight * (b - a)
        // This is equivalent to: output = (1 - weight) * a + weight * b
        // But we'll compute it directly for better numerical stability

        let (a_real, a_imag) = Self::to_split_complex(a.as_slice().unwrap());
        let (b_real, b_imag) = Self::to_split_complex(b.as_slice().unwrap());
        let (w_real, w_imag) = Self::to_split_complex(weight.as_slice().unwrap());

        let mut out_real = vec![0.0; n];
        let mut out_imag = vec![0.0; n];

        for i in 0..n {
            // diff = b - a
            let diff_real = b_real[i] - a_real[i];
            let diff_imag = b_imag[i] - a_imag[i];

            // weighted_diff = weight * diff (complex multiplication)
            let wd_real = w_real[i] * diff_real - w_imag[i] * diff_imag;
            let wd_imag = w_real[i] * diff_imag + w_imag[i] * diff_real;

            // output = a + weighted_diff
            out_real[i] = a_real[i] + wd_real;
            out_imag[i] = a_imag[i] + wd_imag;
        }

        let result = Self::from_split_complex(&out_real, &out_imag);
        for (i, val) in result.iter().enumerate() {
            output.as_slice_mut().unwrap()[i] = *val;
        }
    }

    fn name(&self) -> &'static str {
        "accelerate"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_split_complex_conversion() {
        let data = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ];

        let (real, imag) = AccelerateBackend::to_split_complex(&data);
        assert_eq!(real, vec![1.0, 3.0, 5.0]);
        assert_eq!(imag, vec![2.0, 4.0, 6.0]);

        let result = AccelerateBackend::from_split_complex(&real, &imag);
        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val, data[i]);
        }
    }

    #[test]
    fn test_fft_roundtrip() {
        let backend = AccelerateBackend::new();

        // Use power-of-2 sizes
        let mut input = Array3::<Complex<f64>>::zeros((4, 4, 4));
        input[[0, 0, 0]] = Complex::new(1.0, 0.0);
        input[[1, 1, 1]] = Complex::new(2.0, 0.0);

        let mut forward = Array3::<Complex<f64>>::zeros((4, 4, 4));
        let mut roundtrip = Array3::<Complex<f64>>::zeros((4, 4, 4));

        backend.fft_3d(&input, &mut forward);
        backend.ifft_3d(&forward, &mut roundtrip);

        // Check that we get back the original (within numerical precision)
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    assert_abs_diff_eq!(
                        roundtrip[[i, j, k]].re,
                        input[[i, j, k]].re,
                        epsilon = 1e-10
                    );
                    assert_abs_diff_eq!(
                        roundtrip[[i, j, k]].im,
                        input[[i, j, k]].im,
                        epsilon = 1e-10
                    );
                }
            }
        }
    }

    #[test]
    fn test_scale_operation() {
        let backend = AccelerateBackend::new();
        let input = Array3::<Complex<f64>>::from_elem((2, 2, 2), Complex::new(1.0, 0.0));
        let mut output = Array3::<Complex<f64>>::zeros((2, 2, 2));

        backend.scale(
            Complex::new(2.0, 0.0),
            &input,
            Some(Complex::new(1.0, 0.0)),
            &mut output,
        );

        assert_abs_diff_eq!(output[[0, 0, 0]].re, 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(output[[0, 0, 0]].im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mix_operation() {
        let backend = AccelerateBackend::new();
        let a = Array3::<Complex<f64>>::from_elem((2, 2, 2), Complex::new(1.0, 0.0));
        let b = Array3::<Complex<f64>>::from_elem((2, 2, 2), Complex::new(2.0, 0.0));
        let mut output = Array3::<Complex<f64>>::zeros((2, 2, 2));

        backend.mix(
            Complex::new(2.0, 0.0),
            &a,
            Complex::new(3.0, 0.0),
            &b,
            &mut output,
        );

        assert_abs_diff_eq!(output[[0, 0, 0]].re, 8.0, epsilon = 1e-10);
        assert_abs_diff_eq!(output[[0, 0, 0]].im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lerp_operation() {
        let backend = AccelerateBackend::new();
        let a = Array3::<Complex<f64>>::from_elem((2, 2, 2), Complex::new(0.0, 0.0));
        let b = Array3::<Complex<f64>>::from_elem((2, 2, 2), Complex::new(10.0, 0.0));
        let weight = Array3::<Complex<f64>>::from_elem((2, 2, 2), Complex::new(0.5, 0.0));
        let mut output = Array3::<Complex<f64>>::zeros((2, 2, 2));

        backend.lerp(&a, &b, &weight, &mut output);

        assert_abs_diff_eq!(output[[0, 0, 0]].re, 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(output[[0, 0, 0]].im, 0.0, epsilon = 1e-10);
    }
}
