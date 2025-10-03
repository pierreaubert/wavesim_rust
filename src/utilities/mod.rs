//! Utility functions for wave simulations

pub mod analytical;

use crate::engine::array::{Complex64, WaveArray};
use ndarray::{Array1, Array3};
use num_complex::Complex;
use std::f64::consts::PI;

/// Add absorbing boundaries to a permittivity array
pub fn add_absorbing_boundaries(
    permittivity: WaveArray<Complex64>,
    boundary_widths: [[usize; 2]; 3],
    strength: f64,
    periodic: [bool; 3],
) -> (WaveArray<Complex64>, [(usize, usize); 3]) {
    let shape = permittivity.shape_tuple();

    // Calculate new shape with boundaries
    let new_shape = (
        shape.0 + boundary_widths[0][0] + boundary_widths[0][1],
        shape.1 + boundary_widths[1][0] + boundary_widths[1][1],
        shape.2 + boundary_widths[2][0] + boundary_widths[2][1],
    );

    // Create padded array
    let mut padded = WaveArray::zeros(new_shape);

    // Copy original data to center
    let offset = [
        boundary_widths[0][0],
        boundary_widths[1][0],
        boundary_widths[2][0],
    ];

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                padded.data[[offset[0] + i, offset[1] + j, offset[2] + k]] =
                    permittivity.data[[i, j, k]];
            }
        }
    }

    // Add absorbing boundaries with linear ramp
    for dim in 0..3 {
        if periodic[dim] {
            continue; // Skip periodic boundaries
        }

        // Left boundary
        let left_width = boundary_widths[dim][0];
        if left_width > 0 {
            add_absorption_ramp(&mut padded, dim, 0, left_width, strength, false);
        }

        // Right boundary
        let right_width = boundary_widths[dim][1];
        if right_width > 0 {
            let start = match dim {
                0 => new_shape.0 - right_width,
                1 => new_shape.1 - right_width,
                2 => new_shape.2 - right_width,
                _ => unreachable!(),
            };
            add_absorption_ramp(&mut padded, dim, start, right_width, strength, true);
        }
    }

    // Return padded array and ROI for extracting original region
    let roi = [
        (boundary_widths[0][0], shape.0 + boundary_widths[0][0]),
        (boundary_widths[1][0], shape.1 + boundary_widths[1][0]),
        (boundary_widths[2][0], shape.2 + boundary_widths[2][0]),
    ];

    (padded, roi)
}

/// Helper function to add absorption ramp
fn add_absorption_ramp(
    array: &mut WaveArray<Complex64>,
    dim: usize,
    start: usize,
    width: usize,
    strength: f64,
    reverse: bool,
) {
    let shape = array.shape_tuple();

    for offset in 0..width {
        // Linear ramp profile
        let profile = if reverse {
            (offset as f64 + 0.79) / (width as f64 + 0.66)
        } else {
            (width as f64 - offset as f64 - 0.21) / (width as f64 + 0.66)
        };

        let absorption = Complex::new(0.0, strength * profile);

        match dim {
            0 => {
                for j in 0..shape.1 {
                    for k in 0..shape.2 {
                        array.data[[start + offset, j, k]] += absorption;
                    }
                }
            }
            1 => {
                for i in 0..shape.0 {
                    for k in 0..shape.2 {
                        array.data[[i, start + offset, k]] += absorption;
                    }
                }
            }
            2 => {
                for i in 0..shape.0 {
                    for j in 0..shape.1 {
                        array.data[[i, j, start + offset]] += absorption;
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}

/// Create a point source
pub fn create_source(
    position: [usize; 3],
    _pixel_size: f64,
    _wavelength: f64,
    amplitude: Complex64,
) -> (WaveArray<Complex64>, [usize; 3]) {
    // Create a single-pixel point source
    let mut source = WaveArray::zeros((1, 1, 1));
    source.data[[0, 0, 0]] = amplitude;

    (source, position)
}

/// Create a Gaussian source
pub fn create_gaussian_source(
    position: [usize; 3],
    width: [f64; 3],
    pixel_size: f64,
    _wavelength: f64,
    amplitude: Complex64,
) -> (WaveArray<Complex64>, [usize; 3]) {
    // Calculate source size in pixels
    let size = [
        (4.0 * width[0] / pixel_size).ceil() as usize | 1, // Ensure odd
        (4.0 * width[1] / pixel_size).ceil() as usize | 1,
        (4.0 * width[2] / pixel_size).ceil() as usize | 1,
    ];

    let mut source = WaveArray::zeros((size[0], size[1], size[2]));

    let center = [
        (size[0] / 2) as f64,
        (size[1] / 2) as f64,
        (size[2] / 2) as f64,
    ];

    for i in 0..size[0] {
        for j in 0..size[1] {
            for k in 0..size[2] {
                let dx = (i as f64 - center[0]) * pixel_size;
                let dy = (j as f64 - center[1]) * pixel_size;
                let dz = (k as f64 - center[2]) * pixel_size;

                let r2 =
                    (dx / width[0]).powi(2) + (dy / width[1]).powi(2) + (dz / width[2]).powi(2);

                source.data[[i, j, k]] = amplitude * (-r2).exp();
            }
        }
    }

    // Adjust position to account for source size
    let adjusted_pos = [
        position[0].saturating_sub(size[0] / 2),
        position[1].saturating_sub(size[1] / 2),
        position[2].saturating_sub(size[2] / 2),
    ];

    (source, adjusted_pos)
}

/// Compute the 1D Laplace kernel for finite differences
pub fn laplace_kernel_1d(pixel_size: f64, length: usize) -> Array1<Complex64> {
    if length == 1 {
        return Array1::zeros(1);
    }

    // Frequency coordinates
    let mut x = Array1::zeros(length);
    for i in 0..length {
        x[i] = if i <= length / 2 {
            i as f64 * PI * length as f64 / length as f64
        } else {
            -((length - i) as f64) * PI * length as f64 / length as f64
        };
    }

    // Compute kernel
    let mut kernel = Array1::zeros(length);
    for i in 0..length {
        if i == 0 {
            kernel[i] = Complex::new(PI.powi(2) / (3.0 * pixel_size.powi(2)), 0.0);
        } else {
            let xi = x[i];
            kernel[i] = Complex::new(
                -PI.powi(2) / pixel_size.powi(2) * 2.0 * xi.cos() / xi.powi(2),
                0.0,
            );
        }
    }

    // Adjust to ensure zero average
    let sum: Complex64 = kernel.iter().sum();
    if length.is_multiple_of(2) {
        kernel[length / 2] -= sum;
    } else {
        kernel[length / 2] -= sum / 2.0;
        kernel[length / 2 + 1] -= sum / 2.0;
    }

    kernel
}

/// Normalize an array to a given range
pub fn normalize(
    data: &Array3<f64>,
    min_val: Option<f64>,
    max_val: Option<f64>,
    a: f64,
    b: f64,
) -> Array3<f64> {
    let min = min_val.unwrap_or_else(|| data.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
    let max = max_val.unwrap_or_else(|| data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

    let range = max - min;
    if range.abs() < 1e-10 {
        return Array3::from_elem(data.dim(), a);
    }

    data.map(|&x| (x - min) / range * (b - a) + a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_point_source() {
        let (source, pos) = create_source([10, 10, 10], 0.1, 0.5, Complex64::new(1.0, 0.0));

        assert_eq!(source.shape(), &[1, 1, 1]);
        assert_eq!(source.data[[0, 0, 0]], Complex64::new(1.0, 0.0));
        assert_eq!(pos, [10, 10, 10]);
    }

    #[test]
    fn test_gaussian_source() {
        let (source, pos) = create_gaussian_source(
            [20, 20, 20],
            [1.0, 1.0, 1.0],
            0.1,
            0.5,
            Complex64::new(1.0, 0.0),
        );

        // Check that center has maximum value
        let shape = source.shape();
        let center = [shape[0] / 2, shape[1] / 2, shape[2] / 2];
        let center_val = source.data[[center[0], center[1], center[2]]].norm();

        // Check that values decrease away from center
        let edge_val = source.data[[0, 0, 0]].norm();
        assert!(center_val > edge_val);
    }

    #[test]
    fn test_laplace_kernel() {
        let kernel = laplace_kernel_1d(0.1, 10);

        // Check that kernel has zero average
        let sum: Complex64 = kernel.iter().sum();
        assert_abs_diff_eq!(sum.norm(), 0.0, epsilon = 1e-10);
    }
}
