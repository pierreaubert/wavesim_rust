//! RustFFT backend implementation
//!
//! Pure Rust implementation using the rustfft library.
//! This backend is available on all platforms.

use super::ComputeBackend;
use ndarray::{Array3, Zip};
use num_complex::Complex;
use rustfft::FftPlanner;

/// RustFFT-based compute backend
pub struct RustFFTBackend {
    // RustFFT planner is created on-demand for thread safety
}

impl RustFFTBackend {
    /// Create a new RustFFT backend
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for RustFFTBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for RustFFTBackend {
    fn fft_3d(&self, input: &Array3<Complex<f64>>, output: &mut Array3<Complex<f64>>) {
        let shape = input.shape();
        output.assign(input);

        let mut planner = FftPlanner::new();

        // FFT along axis 0
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let mut slice: Vec<Complex<f64>> =
                    (0..shape[0]).map(|i| output[[i, j, k]]).collect();

                let fft = planner.plan_fft_forward(shape[0]);
                fft.process(&mut slice);

                for (i, val) in slice.iter().enumerate() {
                    output[[i, j, k]] = *val;
                }
            }
        }

        // FFT along axis 1
        for i in 0..shape[0] {
            for k in 0..shape[2] {
                let mut slice: Vec<Complex<f64>> =
                    (0..shape[1]).map(|j| output[[i, j, k]]).collect();

                let fft = planner.plan_fft_forward(shape[1]);
                fft.process(&mut slice);

                for (j, val) in slice.iter().enumerate() {
                    output[[i, j, k]] = *val;
                }
            }
        }

        // FFT along axis 2
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let mut slice: Vec<Complex<f64>> =
                    (0..shape[2]).map(|k| output[[i, j, k]]).collect();

                let fft = planner.plan_fft_forward(shape[2]);
                fft.process(&mut slice);

                for (k, val) in slice.iter().enumerate() {
                    output[[i, j, k]] = *val;
                }
            }
        }
    }

    fn ifft_3d(&self, input: &Array3<Complex<f64>>, output: &mut Array3<Complex<f64>>) {
        let shape = input.shape();
        output.assign(input);

        let mut planner = FftPlanner::new();
        let normalization = 1.0 / (shape[0] * shape[1] * shape[2]) as f64;

        // IFFT along axis 0
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let mut slice: Vec<Complex<f64>> =
                    (0..shape[0]).map(|i| output[[i, j, k]]).collect();

                let fft = planner.plan_fft_inverse(shape[0]);
                fft.process(&mut slice);

                for (i, val) in slice.iter().enumerate() {
                    output[[i, j, k]] = *val;
                }
            }
        }

        // IFFT along axis 1
        for i in 0..shape[0] {
            for k in 0..shape[2] {
                let mut slice: Vec<Complex<f64>> =
                    (0..shape[1]).map(|j| output[[i, j, k]]).collect();

                let fft = planner.plan_fft_inverse(shape[1]);
                fft.process(&mut slice);

                for (j, val) in slice.iter().enumerate() {
                    output[[i, j, k]] = *val;
                }
            }
        }

        // IFFT along axis 2
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let mut slice: Vec<Complex<f64>> =
                    (0..shape[2]).map(|k| output[[i, j, k]]).collect();

                let fft = planner.plan_fft_inverse(shape[2]);
                fft.process(&mut slice);

                for (k, val) in slice.iter().enumerate() {
                    output[[i, j, k]] = *val * normalization;
                }
            }
        }
    }

    fn scale(
        &self,
        scale: Complex<f64>,
        input: &Array3<Complex<f64>>,
        offset: Option<Complex<f64>>,
        output: &mut Array3<Complex<f64>>,
    ) {
        if let Some(off) = offset {
            Zip::from(output)
                .and(input)
                .for_each(|o, &i| *o = scale * i + off);
        } else {
            Zip::from(output)
                .and(input)
                .for_each(|o, &i| *o = scale * i);
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
        Zip::from(output)
            .and(a)
            .and(b)
            .for_each(|o, &a_val, &b_val| *o = alpha * a_val + beta * b_val);
    }

    fn lerp(
        &self,
        a: &Array3<Complex<f64>>,
        b: &Array3<Complex<f64>>,
        weight: &Array3<Complex<f64>>,
        output: &mut Array3<Complex<f64>>,
    ) {
        Zip::from(output)
            .and(a)
            .and(b)
            .and(weight)
            .for_each(|o, &a_val, &b_val, &w| *o = a_val + w * (b_val - a_val));
    }

    fn name(&self) -> &'static str {
        "rustfft"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_fft_roundtrip() {
        let backend = RustFFTBackend::new();
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
        let backend = RustFFTBackend::new();
        let input = Array3::<Complex<f64>>::from_elem((2, 2, 2), Complex::new(1.0, 0.0));
        let mut output = Array3::<Complex<f64>>::zeros((2, 2, 2));

        backend.scale(
            Complex::new(2.0, 0.0),
            &input,
            Some(Complex::new(1.0, 0.0)),
            &mut output,
        );

        assert_eq!(output[[0, 0, 0]], Complex::new(3.0, 0.0));
    }

    #[test]
    fn test_mix_operation() {
        let backend = RustFFTBackend::new();
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

        assert_eq!(output[[0, 0, 0]], Complex::new(8.0, 0.0));
    }

    #[test]
    fn test_lerp_operation() {
        let backend = RustFFTBackend::new();
        let a = Array3::<Complex<f64>>::from_elem((2, 2, 2), Complex::new(0.0, 0.0));
        let b = Array3::<Complex<f64>>::from_elem((2, 2, 2), Complex::new(10.0, 0.0));
        let weight = Array3::<Complex<f64>>::from_elem((2, 2, 2), Complex::new(0.5, 0.0));
        let mut output = Array3::<Complex<f64>>::zeros((2, 2, 2));

        backend.lerp(&a, &b, &weight, &mut output);

        assert_eq!(output[[0, 0, 0]], Complex::new(5.0, 0.0));
    }
}
