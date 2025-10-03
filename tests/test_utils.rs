//! Test utilities for WaveSim tests
//!
//! Provides helper functions for testing similar to the Python test suite

use ndarray::{Array3, ArrayView3};
use num_complex::Complex;
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use wavesim::domain::domain_trait::Domain;
use wavesim::engine::array::{Complex64, WaveArray};

/// Generate a random complex vector for testing
pub fn random_vector(shape: (usize, usize, usize)) -> WaveArray<Complex64> {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 0.5_f64.sqrt()).unwrap();

    let mut data = Array3::zeros(shape);
    for elem in data.iter_mut() {
        let real: f64 = rng.sample(normal);
        let imag: f64 = rng.sample(normal);
        *elem = Complex::new(real, imag);
    }

    WaveArray { data }
}

/// Generate a random permittivity (n²) between 1 and 4 with small positive imaginary part
pub fn random_permittivity(shape: (usize, usize, usize)) -> WaveArray<Complex64> {
    let mut rng = thread_rng();
    let real_dist = Uniform::new(1.0, 4.0);
    let imag_dist = Uniform::new(0.0, 0.4);

    let mut data = Array3::zeros(shape);
    for elem in data.iter_mut() {
        let real: f64 = rng.sample(real_dist);
        let imag: f64 = rng.sample(imag_dist);
        *elem = Complex::new(real, imag);
    }

    WaveArray { data }
}

/// Check if two arrays are close to each other
pub fn all_close(a: &WaveArray<Complex64>, b: &WaveArray<Complex64>, rtol: f64, atol: f64) -> bool {
    if a.shape() != b.shape() {
        println!("Shapes do not match: {:?} != {:?}", a.shape(), b.shape());
        return false;
    }

    for (a_val, b_val) in a.data.iter().zip(b.data.iter()) {
        let diff = (a_val - b_val).norm();
        let tolerance = atol + rtol * a_val.norm().max(b_val.norm());

        if diff > tolerance {
            println!(
                "Values differ: {} vs {}, diff = {}, tolerance = {}",
                a_val, b_val, diff, tolerance
            );
            return false;
        }
    }

    true
}

/// Compute relative error between two fields
pub fn relative_error(computed: &WaveArray<Complex64>, reference: &WaveArray<Complex64>) -> f64 {
    let mut error_sum = 0.0;
    let mut ref_sum = 0.0;

    for (comp, ref_val) in computed.data.iter().zip(reference.data.iter()) {
        let diff = comp - ref_val;
        error_sum += diff.norm_sqr();
        ref_sum += ref_val.norm_sqr();
    }

    if ref_sum == 0.0 {
        if error_sum == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (error_sum / ref_sum).sqrt()
    }
}

/// Maximum absolute error
pub fn max_abs_error(computed: &WaveArray<Complex64>, reference: &WaveArray<Complex64>) -> f64 {
    let mut max_error: f64 = 0.0;

    for (comp, ref_val) in computed.data.iter().zip(reference.data.iter()) {
        let error = (comp - ref_val).norm();
        max_error = f64::max(max_error, error);
    }

    max_error
}

/// Maximum relative error (normalized by RMS of reference)
pub fn max_relative_error(
    computed: &WaveArray<Complex64>,
    reference: &WaveArray<Complex64>,
) -> f64 {
    let max_err = max_abs_error(computed, reference);
    let ref_rms = (reference.norm_squared() / reference.data.len() as f64).sqrt();

    if ref_rms == 0.0 {
        if max_err == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        max_err / ref_rms
    }
}

/// Generate analytical solution for 1D free space propagation
/// Based on the analytical solution from the Python test suite
/// This is the solution for a point source in a 1D homogeneous medium
pub fn analytical_solution_1d(x: &[f64], wavelength: f64) -> WaveArray<Complex64> {
    let n = x.len();
    if n < 2 {
        return WaveArray {
            data: Array3::zeros((n, 1, 1)),
        };
    }

    // Get pixel size from spacing
    let h = (x[1] - x[0]).abs();
    let k = 2.0 * std::f64::consts::PI / wavelength;

    let mut result = Array3::zeros((n, 1, 1));

    for (i, &xi) in x.iter().enumerate() {
        let x_abs = xi.abs();

        if x_abs < 1e-10 {
            // Special case at source point
            // u(0) = i*h/(2k) * (1 + 2i*arctanh(hk/π)/π)
            let hk_pi = h * k / std::f64::consts::PI;
            let arctanh_term = hk_pi.atanh();
            let value = Complex::new(0.0, h / (2.0 * k))
                * Complex::new(1.0, 2.0 * arctanh_term / std::f64::consts::PI);
            result[[i, 0, 0]] = value;
        } else {
            // Away from source: simplified version without exponential integrals
            // For testing purposes, use a simpler Green's function approximation
            // This matches the expected field pattern for 1D propagation
            let phi = k * x_abs;

            // Simplified 1D Green's function
            // G(x) ≈ (i/(2k)) * exp(i*k*|x|)
            let exp_ikx = Complex::new(phi.cos(), phi.sin());
            result[[i, 0, 0]] = Complex::new(0.0, h / (2.0 * k)) * exp_ikx;
        }
    }

    WaveArray { data: result }
}

/// Create a point source at specified position
pub fn create_point_source(
    position: [usize; 3],
    shape: (usize, usize, usize),
    amplitude: Complex64,
) -> WaveArray<Complex64> {
    let mut source = WaveArray::zeros(shape);
    if position[0] < shape.0 && position[1] < shape.1 && position[2] < shape.2 {
        source.data[[position[0], position[1], position[2]]] = amplitude;
    }
    source
}

/// Assert that values are approximately equal with custom message
#[macro_export]
macro_rules! assert_close {
    ($left:expr, $right:expr, $tolerance:expr) => {
        {
            let left_val = $left;
            let right_val = $right;
            let tol = $tolerance;
            let diff = (left_val - right_val).abs();
            assert!(
                diff < tol,
                "assertion failed: `(left ≈ right)`\n  left: `{:?}`,\n right: `{:?}`,\n  diff: `{:?}`,\n   tol: `{:?}`",
                left_val, right_val, diff, tol
            );
        }
    };
    ($left:expr, $right:expr, $tolerance:expr, $($arg:tt)+) => {
        {
            let left_val = $left;
            let right_val = $right;
            let tol = $tolerance;
            let diff = (left_val - right_val).abs();
            assert!(
                diff < tol,
                "assertion failed: `(left ≈ right)`: {}\n  left: `{:?}`,\n right: `{:?}`,\n  diff: `{:?}`,\n   tol: `{:?}`",
                format_args!($($arg)+), left_val, right_val, diff, tol
            );
        }
    };
}
