//! Helmholtz equation solver implementation
//!
//! Solves the Helmholtz equation: (∇² + k²)u = f
//! using the Modified Born Series approach

use crate::domain::domain_trait::Domain;
use crate::engine::array::{Complex64, WaveArray};
use crate::engine::operations::{fft_3d, ifft_3d, multiply};
use crate::utilities::laplace_kernel_1d;
use ndarray::Array1;
use num_complex::Complex;
use num_traits::Zero;
use std::f64::consts::PI;

/// Helmholtz domain solver
#[derive(Debug)]
pub struct HelmholtzDomain {
    /// Permittivity (refractive index squared) with boundary conditions applied
    pub permittivity: WaveArray<Complex64>,
    /// The shape of the domain
    pub shape: (usize, usize, usize),
    /// Pixel size in micrometers
    pub pixel_size: f64,
    /// Wavelength in micrometers
    pub wavelength: f64,
    /// Wave number squared (k₀²)
    pub k02: Complex64,
    /// Laplace kernels for each dimension
    pub laplace_kernels: [WaveArray<Complex64>; 3],
    /// The scattering potential B = 1 - V
    pub b_scat: WaveArray<Complex64>,
    /// Scaling factor for convergence
    pub scale_factor: Complex64,
    /// Optimal shift for reducing ||V||
    pub shift: Complex64,
    /// Periodic boundary conditions
    pub periodic: [bool; 3],
    /// Boundary widths
    pub boundary_widths: [[usize; 2]; 3],
    /// Whether to use domain decomposition
    pub use_blocks: bool,
    /// Number of domains for decomposition
    pub n_domains: Option<[usize; 3]>,
}

impl HelmholtzDomain {
    /// Create a new Helmholtz domain
    pub fn new(
        permittivity: WaveArray<Complex64>,
        pixel_size: f64,
        wavelength: f64,
        periodic: [bool; 3],
        boundary_widths: [[usize; 2]; 3],
    ) -> Self {
        let shape = permittivity.shape_tuple();

        // Calculate wave number
        let k0 = 2.0 * PI / wavelength;
        let k02 = Complex::new(k0 * k0, 0.0);

        // Create Laplace kernels for each dimension
        let laplace_kernels = Self::create_laplace_kernels(shape, pixel_size, periodic);

        // Calculate optimal shift and scaling
        let (shift, scale_factor) = Self::calculate_scaling(&permittivity, k02);

        // Create B_scat = 1 - V_scat
        let mut b_scat = permittivity.clone();
        Self::prepare_scattering_potential(&mut b_scat, k02, shift, scale_factor);

        Self {
            permittivity,
            shape,
            pixel_size,
            wavelength,
            k02,
            laplace_kernels,
            b_scat,
            scale_factor,
            shift,
            periodic,
            boundary_widths,
            use_blocks: false,
            n_domains: None,
        }
    }

    /// Enable domain decomposition
    pub fn with_domain_decomposition(mut self, n_domains: [usize; 3]) -> Self {
        self.use_blocks = true;
        self.n_domains = Some(n_domains);
        self
    }

    /// Create Laplace kernels for finite difference operators
    fn create_laplace_kernels(
        shape: (usize, usize, usize),
        pixel_size: f64,
        periodic: [bool; 3],
    ) -> [WaveArray<Complex64>; 3] {
        let mut kernels = [
            WaveArray::zeros(shape),
            WaveArray::zeros(shape),
            WaveArray::zeros(shape),
        ];

        // Create 1D kernels and expand to 3D
        for dim in 0..3 {
            let length = match dim {
                0 => shape.0,
                1 => shape.1,
                2 => shape.2,
                _ => unreachable!(),
            };

            let kernel_1d = if periodic[dim] {
                Self::periodic_laplace_kernel(length, pixel_size)
            } else {
                laplace_kernel_1d(pixel_size, length)
            };

            // Expand 1D kernel to 3D
            kernels[dim] = Self::expand_kernel_to_3d(kernel_1d, shape, dim);
        }

        kernels
    }

    /// Create periodic Laplace kernel using FFT frequency convention
    ///
    /// This computes the Fourier-space representation of the Laplace operator
    /// for periodic boundaries, matching the Python implementation.
    ///
    /// The kernel is computed as: `kernel[i] = -k_i^2`
    /// where `k_i` is the i-th FFT frequency following NumPy's `fftfreq` convention.
    ///
    /// # Arguments
    /// * `length` - The number of points in the 1D kernel
    /// * `pixel_size` - The spatial discretization step size
    ///
    /// # Returns
    /// A 1D array of complex numbers representing the Laplace kernel in Fourier space
    ///
    /// # Reference
    /// Python implementation: `kernel_1d = -np.fft.fftfreq(length, d=pixel_size / (2.0 * np.pi)) ** 2`
    fn periodic_laplace_kernel(length: usize, pixel_size: f64) -> Array1<Complex64> {
        let mut kernel = Array1::zeros(length);

        // d parameter matching Python's fftfreq convention
        let d = pixel_size / (2.0 * PI);

        for i in 0..length {
            // FFT frequency convention: [0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n)
            let freq = if i <= length / 2 {
                i as f64 / (d * length as f64)
            } else {
                (i as f64 - length as f64) / (d * length as f64)
            };

            // Laplace kernel is the negative square of the frequency
            kernel[i] = Complex::new(-freq * freq, 0.0);
        }

        kernel
    }

    /// Expand 1D kernel to 3D array along specified dimension
    fn expand_kernel_to_3d(
        kernel_1d: Array1<Complex64>,
        shape: (usize, usize, usize),
        dim: usize,
    ) -> WaveArray<Complex64> {
        let mut kernel_3d = WaveArray::zeros(shape);

        match dim {
            0 => {
                for i in 0..shape.0 {
                    for j in 0..shape.1 {
                        for k in 0..shape.2 {
                            kernel_3d.data[[i, j, k]] = kernel_1d[i];
                        }
                    }
                }
            }
            1 => {
                for i in 0..shape.0 {
                    for j in 0..shape.1 {
                        for k in 0..shape.2 {
                            kernel_3d.data[[i, j, k]] = kernel_1d[j];
                        }
                    }
                }
            }
            2 => {
                for i in 0..shape.0 {
                    for j in 0..shape.1 {
                        for k in 0..shape.2 {
                            kernel_3d.data[[i, j, k]] = kernel_1d[k];
                        }
                    }
                }
            }
            _ => unreachable!(),
        }

        kernel_3d
    }

    /// Calculate optimal scaling parameters
    fn calculate_scaling(
        permittivity: &WaveArray<Complex64>,
        k02: Complex64,
    ) -> (Complex64, Complex64) {
        // Find the enclosing circle of the permittivity values
        let (center, radius) = Self::enclosing_circle(permittivity);

        // Optimal shift to minimize ||V||
        let shift = k02 * center;

        // Scale factor to ensure convergence
        // We want |scale * k02 * radius| < 1 for convergence
        // So |scale| < 1 / (|k02| * radius)
        // Use 0.95 safety factor
        let scale_factor = if radius < 1e-10 {
            // For homogeneous medium, we don't need Born series
            // Use a small imaginary scale to avoid division by zero
            Complex::new(0.0, -0.01)
        } else {
            // Ensure the scattering potential norm is bounded
            // We want ||scale * k02 * (permittivity - center)|| < 0.95
            let max_scale = 0.95 / (radius * k02.norm());
            // Use imaginary unit for optimal convergence
            Complex::new(0.0, -max_scale.min(1.0)) // Cap at 1.0 to avoid huge scale factors
        };

        (shift, scale_factor)
    }

    /// Find enclosing circle for permittivity values
    fn enclosing_circle(permittivity: &WaveArray<Complex64>) -> (Complex64, f64) {
        let mut r_min = f64::INFINITY;
        let mut r_max = f64::NEG_INFINITY;
        let mut i_min = f64::INFINITY;
        let mut i_max = f64::NEG_INFINITY;

        for val in permittivity.data.iter() {
            r_min = r_min.min(val.re);
            r_max = r_max.max(val.re);
            i_min = i_min.min(val.im);
            i_max = i_max.max(val.im);
        }

        let center = Complex::new((r_min + r_max) / 2.0, (i_min + i_max) / 2.0);

        let mut radius: f64 = 0.0;
        for val in permittivity.data.iter() {
            let dist = (val - center).norm();
            radius = f64::max(radius, dist);
        }

        (center, radius)
    }

    /// Prepare the scattering potential B = 1 - V
    pub fn prepare_scattering_potential(
        b_scat: &mut WaveArray<Complex64>,
        k02: Complex64,
        shift: Complex64,
        scale_factor: Complex64,
    ) {
        // B = scale * shift + 1 - scale * k02 * permittivity
        //   = 1 + scale * (shift - k02 * permittivity)
        //   = 1 - scale * k02 * (permittivity - shift/k02)

        let shift_normalized = shift / k02;

        for val in b_scat.data.iter_mut() {
            *val = Complex64::new(1.0, 0.0) - scale_factor * k02 * (*val - shift_normalized);
        }
    }

    /// Apply the Laplace operator in Fourier space
    fn apply_laplace(&self, x: &WaveArray<Complex64>, out: &mut WaveArray<Complex64>) {
        // Transform to Fourier space
        let mut x_fft = x.clone();
        fft_3d(x, &mut x_fft);

        // Build the operator (L+1) in Fourier space
        let mut operator = WaveArray::zeros(self.shape);
        for kernel in &self.laplace_kernels {
            operator.data += &kernel.data;
        }

        // Add shift and scale: (scale * L + scale * shift + 1)
        let shift_scaled = self.scale_factor * self.shift + Complex64::new(1.0, 0.0);
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                for k in 0..self.shape.2 {
                    let laplace_val = operator.data[[i, j, k]];
                    // Apply operator: (scale * L + 1) * x_fft
                    x_fft.data[[i, j, k]] *= laplace_val * self.scale_factor + shift_scaled;
                }
            }
        }

        // Transform back to real space
        ifft_3d(&x_fft, out);
    }

    /// Apply the inverse Laplace operator (L+1)^{-1} using FFT
    fn apply_inverse_laplace(&self, x: &WaveArray<Complex64>, out: &mut WaveArray<Complex64>) {
        // Transform to Fourier space
        let mut x_fft = x.clone();
        fft_3d(x, &mut x_fft);

        // Create the operator (L+1) in Fourier space
        let mut operator = WaveArray::zeros(self.shape);
        for kernel in &self.laplace_kernels {
            operator.data += &kernel.data;
        }

        // Add shift and scale
        let shift_scaled = self.scale_factor * self.shift + Complex64::new(1.0, 0.0);
        for val in operator.data.iter_mut() {
            *val = *val * self.scale_factor + shift_scaled;
        }

        // Invert and multiply
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                for k in 0..self.shape.2 {
                    let op_val = operator.data[[i, j, k]];
                    if op_val.norm() > 1e-10 {
                        x_fft.data[[i, j, k]] /= op_val;
                    } else {
                        x_fft.data[[i, j, k]] = Complex::zero();
                    }
                }
            }
        }

        // Transform back to real space
        ifft_3d(&x_fft, out);
    }
}

impl Domain for HelmholtzDomain {
    fn medium(&self, x: &WaveArray<Complex64>, out: &mut WaveArray<Complex64>) {
        // Apply B = 1 - V
        *out = multiply(x, &self.b_scat);
    }

    fn propagator(&self, x: &WaveArray<Complex64>, out: &mut WaveArray<Complex64>) {
        // Apply (L+1)^{-1}
        self.apply_inverse_laplace(x, out);
    }

    fn inverse_propagator(&self, x: &WaveArray<Complex64>, out: &mut WaveArray<Complex64>) {
        // Apply (L+1)
        self.apply_laplace(x, out);
    }

    fn shape(&self) -> (usize, usize, usize) {
        self.shape
    }

    fn scale(&self) -> Complex64 {
        self.scale_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_helmholtz_creation() {
        let permittivity = WaveArray::from_scalar((10, 10, 10), Complex::new(1.0, 0.0));
        let domain = HelmholtzDomain::new(
            permittivity,
            0.1, // pixel_size
            0.5, // wavelength
            [false, false, false],
            [[1, 1], [1, 1], [1, 1]],
        );

        assert_eq!(domain.shape, (10, 10, 10));
        assert_eq!(domain.wavelength, 0.5);
        assert_eq!(domain.pixel_size, 0.1);
    }

    #[test]
    fn test_laplace_kernel_creation() {
        let kernels =
            HelmholtzDomain::create_laplace_kernels((10, 10, 10), 0.1, [false, false, false]);

        assert_eq!(kernels.len(), 3);
        for kernel in &kernels {
            assert_eq!(kernel.shape(), &[10, 10, 10]);
        }
    }

    #[test]
    fn test_scaling_calculation() {
        let mut permittivity = WaveArray::from_scalar((5, 5, 5), Complex::new(1.0, 0.0));
        permittivity.data[[2, 2, 2]] = Complex::new(1.5, 0.1);

        let k02 = Complex::new(1.0, 0.0);
        let (shift, scale) = HelmholtzDomain::calculate_scaling(&permittivity, k02);

        // Check that we get reasonable values
        assert!(shift.norm() > 0.0);
        assert!(scale.norm() > 0.0);
    }

    #[test]
    fn test_periodic_laplace_kernel() {
        use approx::assert_abs_diff_eq;

        // Test with length=8, pixel_size=1.0 (simple test case)
        let length = 8;
        let pixel_size = 1.0;
        let kernel = HelmholtzDomain::periodic_laplace_kernel(length, pixel_size);

        // The d parameter in Python's fftfreq convention
        let d = pixel_size / (2.0 * PI);

        // Verify kernel values match expected FFT frequencies squared and negated
        // For length=8, frequencies should be: [0, 1, 2, 3, -4, -3, -2, -1] / (d*8)
        let expected_freqs = vec![0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0];

        for (i, &expected_freq) in expected_freqs.iter().enumerate() {
            let freq = expected_freq / (d * length as f64);
            let expected_kernel_val = -freq * freq;
            assert_abs_diff_eq!(kernel[i].re, expected_kernel_val, epsilon = 1e-10);
            assert_abs_diff_eq!(kernel[i].im, 0.0, epsilon = 1e-10);
        }

        // First element should always be zero (DC component)
        assert_abs_diff_eq!(kernel[0].re, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(kernel[0].im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_periodic_laplace_kernel_odd_length() {
        use approx::assert_abs_diff_eq;

        // Test with odd length
        let length = 9;
        let pixel_size = 0.5;
        let kernel = HelmholtzDomain::periodic_laplace_kernel(length, pixel_size);

        let d = pixel_size / (2.0 * PI);

        // For length=9, frequencies should be: [0, 1, 2, 3, 4, -4, -3, -2, -1] / (d*9)
        let expected_freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, -4.0, -3.0, -2.0, -1.0];

        for (i, &expected_freq) in expected_freqs.iter().enumerate() {
            let freq = expected_freq / (d * length as f64);
            let expected_kernel_val = -freq * freq;
            assert_abs_diff_eq!(kernel[i].re, expected_kernel_val, epsilon = 1e-10);
        }

        // First element should be zero
        assert_abs_diff_eq!(kernel[0].re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_periodic_vs_nonperiodic_helmholtz() {
        // Create a simple test case with a homogeneous medium
        let shape = (16, 16, 16);
        let permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
        let pixel_size = 0.1;
        let wavelength = 1.0;

        // Create periodic domain
        let domain_periodic = HelmholtzDomain::new(
            permittivity.clone(),
            pixel_size,
            wavelength,
            [true, true, true],
            [[0, 0], [0, 0], [0, 0]],
        );

        // Create non-periodic domain
        let domain_nonperiodic = HelmholtzDomain::new(
            permittivity.clone(),
            pixel_size,
            wavelength,
            [false, false, false],
            [[2, 2], [2, 2], [2, 2]],
        );

        // Test that kernels are different
        // The key difference is in how the frequency components are computed
        assert!(
            domain_periodic.laplace_kernels[0].data[[1, 0, 0]]
                != domain_nonperiodic.laplace_kernels[0].data[[1, 0, 0]],
            "Periodic and non-periodic kernels should be different"
        );

        // Verify that periodic kernel has the expected structure
        // DC component (0 frequency) should be zero
        assert_eq!(
            domain_periodic.laplace_kernels[0].data[[0, 0, 0]].re,
            domain_periodic.laplace_kernels[1].data[[0, 0, 0]].re,
            "DC components should be equal across dimensions"
        );

        // Test propagator on a simple input
        let mut input = WaveArray::zeros(shape);
        input.data[[8, 8, 8]] = Complex::new(1.0, 0.0); // Point source in center

        let mut output_periodic = WaveArray::zeros(shape);
        let mut output_nonperiodic = WaveArray::zeros(shape);

        domain_periodic.propagator(&input, &mut output_periodic);
        domain_nonperiodic.propagator(&input, &mut output_nonperiodic);

        // Both should produce non-zero output
        assert!(
            output_periodic.data[[8, 8, 8]].norm() > 0.0,
            "Periodic domain propagator should produce non-zero output"
        );
        assert!(
            output_nonperiodic.data[[8, 8, 8]].norm() > 0.0,
            "Non-periodic domain propagator should produce non-zero output"
        );

        // The outputs will differ due to different boundary conditions
        // This is expected behavior, not a bug
        println!(
            "Periodic center value: {}",
            output_periodic.data[[8, 8, 8]].norm()
        );
        println!(
            "Non-periodic center value: {}",
            output_nonperiodic.data[[8, 8, 8]].norm()
        );
        println!("Note: Different values are expected due to different boundary conditions");
    }
}
