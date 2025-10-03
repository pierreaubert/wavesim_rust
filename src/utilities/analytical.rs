//! Analytical solutions for the Helmholtz equation
//!
//! This module provides analytical solutions to the Helmholtz equation ∇²u + k²u = 0
//! for simple geometries like rectangles, circles, and spheres. These solutions are
//! obtained using the method of separation of variables.

use crate::engine::array::{Complex64, WaveArray};
use ndarray::Array2;
use num_complex::Complex;
use num_traits::Zero;
use std::f64::consts::PI;

/// Boundary conditions for analytical solutions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition {
    /// Dirichlet boundary condition (u = 0 on boundary)
    Dirichlet,
    /// Neumann boundary condition (du/dn = 0 on boundary)
    Neumann,
}

/// Rectangle analytical solution parameters
#[derive(Debug, Clone)]
pub struct RectangleParams {
    /// Rectangle dimensions [Lx, Ly, Lz]
    pub dimensions: [f64; 3],
    /// Boundary conditions for each face
    pub boundary_conditions: [BoundaryCondition; 6], // [x_min, x_max, y_min, y_max, z_min, z_max]
    /// Maximum mode numbers [nx_max, ny_max, nz_max]
    pub max_modes: [usize; 3],
}

/// Circle analytical solution parameters  
#[derive(Debug, Clone)]
pub struct CircleParams {
    /// Circle radius
    pub radius: f64,
    /// Boundary condition at r = radius
    pub boundary_condition: BoundaryCondition,
    /// Maximum mode numbers [n_max, m_max] for angular and radial modes
    pub max_modes: [usize; 2],
}

/// Sphere analytical solution parameters
#[derive(Debug, Clone)]
pub struct SphereParams {
    /// Sphere radius
    pub radius: f64,
    /// Boundary condition at r = radius
    pub boundary_condition: BoundaryCondition,
    /// Maximum mode numbers [n_max, m_max] for spherical harmonics and radial modes
    pub max_modes: [usize; 2],
}

/// Analytical solution for rectangular geometry
pub struct RectangularSolution {
    params: RectangleParams,
    eigenvalues: Vec<f64>,
    coefficients: Vec<Complex64>,
}

/// Analytical solution for circular geometry
pub struct CircularSolution {
    params: CircleParams,
    eigenvalues: Vec<f64>,
    coefficients: Vec<Complex64>,
}

/// Analytical solution for spherical geometry
pub struct SphericalSolution {
    params: SphereParams,
    eigenvalues: Vec<f64>,
    coefficients: Vec<Complex64>,
}

impl RectangularSolution {
    /// Create a new rectangular solution
    pub fn new(params: RectangleParams) -> Self {
        let mut eigenvalues = Vec::new();
        let mut coefficients = Vec::new();

        // Generate eigenvalues for the rectangular domain
        for nx in 1..=params.max_modes[0] {
            for ny in 1..=params.max_modes[1] {
                for nz in 1..=params.max_modes[2] {
                    let k_x = nx as f64 * PI / params.dimensions[0];
                    let k_y = ny as f64 * PI / params.dimensions[1];
                    let k_z = nz as f64 * PI / params.dimensions[2];

                    let k_squared = k_x * k_x + k_y * k_y + k_z * k_z;
                    eigenvalues.push(k_squared);

                    // Default coefficient (can be set based on source)
                    coefficients.push(Complex64::new(1.0, 0.0));
                }
            }
        }

        Self {
            params,
            eigenvalues,
            coefficients,
        }
    }

    /// Evaluate the solution at a given point
    pub fn evaluate_at(&self, x: f64, y: f64, z: f64) -> Complex64 {
        let mut result = Complex64::zero();
        let mut idx = 0;

        for nx in 1..=self.params.max_modes[0] {
            for ny in 1..=self.params.max_modes[1] {
                for nz in 1..=self.params.max_modes[2] {
                    let k_x = nx as f64 * PI / self.params.dimensions[0];
                    let k_y = ny as f64 * PI / self.params.dimensions[1];
                    let k_z = nz as f64 * PI / self.params.dimensions[2];

                    let mode_value = (k_x * x).sin() * (k_y * y).sin() * (k_z * z).sin();

                    result += self.coefficients[idx] * mode_value;
                    idx += 1;
                }
            }
        }

        result
    }

    /// Compute the solution on a regular grid
    pub fn evaluate_on_grid(
        &self,
        shape: (usize, usize, usize),
        grid_spacing: [f64; 3],
        offset: [f64; 3],
    ) -> WaveArray<Complex64> {
        let mut field = WaveArray::zeros(shape);

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                for k in 0..shape.2 {
                    let x = offset[0] + i as f64 * grid_spacing[0];
                    let y = offset[1] + j as f64 * grid_spacing[1];
                    let z = offset[2] + k as f64 * grid_spacing[2];

                    field.data[[i, j, k]] = self.evaluate_at(x, y, z);
                }
            }
        }

        field
    }

    /// Get the eigenvalues for this solution
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Set coefficients for the solution
    pub fn set_coefficients(&mut self, coefficients: Vec<Complex64>) {
        assert_eq!(coefficients.len(), self.coefficients.len());
        self.coefficients = coefficients;
    }
}

impl CircularSolution {
    /// Create a new circular solution
    pub fn new(params: CircleParams) -> Self {
        // For now, create a simple implementation
        // In a full implementation, this would compute Bessel function zeros
        let eigenvalues = vec![1.0]; // Placeholder
        let coefficients = vec![Complex64::new(1.0, 0.0)]; // Placeholder

        Self {
            params,
            eigenvalues,
            coefficients,
        }
    }

    /// Evaluate the solution at a given point (r, theta)
    pub fn evaluate_at_polar(&self, r: f64, theta: f64) -> Complex64 {
        // Placeholder implementation
        // Full implementation would use Bessel functions J_n(k*r) and trigonometric functions
        if r <= self.params.radius {
            let k = self.eigenvalues[0].sqrt();
            Complex64::new((k * r).sin() * theta.cos(), 0.0)
        } else {
            Complex64::zero()
        }
    }

    /// Evaluate the solution at Cartesian coordinates
    pub fn evaluate_at(&self, x: f64, y: f64) -> Complex64 {
        let r = (x * x + y * y).sqrt();
        let theta = y.atan2(x);
        self.evaluate_at_polar(r, theta)
    }

    /// Compute the solution on a regular 2D grid
    pub fn evaluate_on_grid_2d(
        &self,
        shape: (usize, usize),
        grid_spacing: [f64; 2],
        center: [f64; 2],
    ) -> Array2<Complex64> {
        let mut field = Array2::zeros(shape);

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                let x = center[0] + (i as f64 - shape.0 as f64 / 2.0) * grid_spacing[0];
                let y = center[1] + (j as f64 - shape.1 as f64 / 2.0) * grid_spacing[1];

                field[[i, j]] = self.evaluate_at(x, y);
            }
        }

        field
    }
}

impl SphericalSolution {
    /// Create a new spherical solution
    pub fn new(params: SphereParams) -> Self {
        // Placeholder implementation
        // In a full implementation, this would compute spherical Bessel function zeros
        let eigenvalues = vec![1.0]; // Placeholder
        let coefficients = vec![Complex64::new(1.0, 0.0)]; // Placeholder

        Self {
            params,
            eigenvalues,
            coefficients,
        }
    }

    /// Evaluate the solution at spherical coordinates (r, theta, phi)
    pub fn evaluate_at_spherical(&self, r: f64, theta: f64, phi: f64) -> Complex64 {
        // Placeholder implementation
        // Full implementation would use spherical Bessel functions j_n(k*r) and spherical harmonics Y_nm(theta, phi)
        if r <= self.params.radius {
            let k = self.eigenvalues[0].sqrt();
            Complex64::new((k * r).sin() * theta.sin() * phi.cos(), 0.0)
        } else {
            Complex64::zero()
        }
    }

    /// Evaluate the solution at Cartesian coordinates
    pub fn evaluate_at(&self, x: f64, y: f64, z: f64) -> Complex64 {
        let r = (x * x + y * y + z * z).sqrt();
        let theta = (z / r).acos();
        let phi = y.atan2(x);
        self.evaluate_at_spherical(r, theta, phi)
    }

    /// Compute the solution on a regular 3D grid
    pub fn evaluate_on_grid(
        &self,
        shape: (usize, usize, usize),
        grid_spacing: [f64; 3],
        center: [f64; 3],
    ) -> WaveArray<Complex64> {
        let mut field = WaveArray::zeros(shape);

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                for k in 0..shape.2 {
                    let x = center[0] + (i as f64 - shape.0 as f64 / 2.0) * grid_spacing[0];
                    let y = center[1] + (j as f64 - shape.1 as f64 / 2.0) * grid_spacing[1];
                    let z = center[2] + (k as f64 - shape.2 as f64 / 2.0) * grid_spacing[2];

                    field.data[[i, j, k]] = self.evaluate_at(x, y, z);
                }
            }
        }

        field
    }
}

/// Compute Bessel function of the first kind J_n(x) using series expansion
/// This is a simplified implementation for small arguments
fn bessel_j(n: usize, x: f64) -> f64 {
    if x.abs() < 1e-10 {
        return if n == 0 { 1.0 } else { 0.0 };
    }

    // Series expansion for small to moderate arguments
    let mut result = 0.0;
    let mut term = if n == 0 {
        1.0
    } else {
        (x / 2.0).powi(n as i32) / factorial(n)
    };

    for k in 0..50 {
        result += term;

        term *= -(x * x / 4.0) / ((k + 1) as f64 * (k + n + 1) as f64);

        if term.abs() < 1e-15 {
            break;
        }
    }

    result
}

/// Compute spherical Bessel function of the first kind j_n(x)
fn spherical_bessel_j(n: usize, x: f64) -> f64 {
    if x.abs() < 1e-10 {
        return if n == 0 { 1.0 } else { 0.0 };
    }

    // j_n(x) = sqrt(π/(2x)) * J_{n+1/2}(x)
    // Simplified implementation for small n
    match n {
        0 => x.sin() / x,
        1 => x.sin() / (x * x) - x.cos() / x,
        2 => (3.0 / (x * x) - 1.0) * x.sin() / x - 3.0 * x.cos() / (x * x),
        _ => {
            // Recursive relation: j_{n+1}(x) = (2n+1)/x * j_n(x) - j_{n-1}(x)
            let mut j_prev = spherical_bessel_j(0, x);
            let mut j_curr = spherical_bessel_j(1, x);

            for k in 2..=n {
                let j_next = (2 * k - 1) as f64 / x * j_curr - j_prev;
                j_prev = j_curr;
                j_curr = j_next;
            }

            j_curr
        }
    }
}

/// Compute spherical harmonic Y_l^m(theta, phi)
/// Simplified implementation for low orders
fn spherical_harmonic(l: usize, m: i32, theta: f64, phi: f64) -> Complex64 {
    // Simplified implementation for common cases
    match (l, m) {
        (0, 0) => Complex64::new(0.5 * (1.0 / PI).sqrt(), 0.0),
        (1, -1) => {
            let factor = 0.5 * (3.0 / (2.0 * PI)).sqrt();
            Complex64::new(0.0, factor * theta.sin()) * Complex64::new((-phi).cos(), (-phi).sin())
        }
        (1, 0) => Complex64::new(0.5 * (3.0 / PI).sqrt() * theta.cos(), 0.0),
        (1, 1) => {
            let factor = -0.5 * (3.0 / (2.0 * PI)).sqrt();
            Complex64::new(0.0, factor * theta.sin()) * Complex64::new(phi.cos(), phi.sin())
        }
        _ => Complex64::new(1.0, 0.0), // Placeholder for higher orders
    }
}

/// Compute factorial
fn factorial(n: usize) -> f64 {
    match n {
        0 | 1 => 1.0,
        2 => 2.0,
        3 => 6.0,
        4 => 24.0,
        5 => 120.0,
        6 => 720.0,
        7 => 5040.0,
        8 => 40320.0,
        _ => {
            let mut result = 1.0;
            for i in 2..=n {
                result *= i as f64;
            }
            result
        }
    }
}

/// Compare numerical and analytical solutions
pub fn compare_solutions(
    numerical: &WaveArray<Complex64>,
    analytical: &WaveArray<Complex64>,
) -> (f64, f64, f64) {
    assert_eq!(numerical.shape(), analytical.shape());

    let mut l2_error: f64 = 0.0;
    let mut max_error: f64 = 0.0;
    let mut analytical_norm: f64 = 0.0;

    for (num_val, ana_val) in numerical.data.iter().zip(analytical.data.iter()) {
        let error = (num_val - ana_val).norm();
        l2_error += error * error;
        max_error = max_error.max(error);
        analytical_norm += ana_val.norm_sqr();
    }

    l2_error = l2_error.sqrt();
    analytical_norm = analytical_norm.sqrt();

    let relative_l2_error = if analytical_norm > 1e-15 {
        l2_error / analytical_norm
    } else {
        l2_error
    };

    (l2_error, max_error, relative_l2_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_rectangular_solution_creation() {
        let params = RectangleParams {
            dimensions: [1.0, 1.0, 1.0],
            boundary_conditions: [BoundaryCondition::Dirichlet; 6],
            max_modes: [2, 2, 2],
        };

        let solution = RectangularSolution::new(params);

        assert_eq!(solution.eigenvalues.len(), 8); // 2×2×2
        assert_eq!(solution.coefficients.len(), 8);
    }

    #[test]
    fn test_rectangular_solution_evaluation() {
        let params = RectangleParams {
            dimensions: [PI, PI, PI],
            boundary_conditions: [BoundaryCondition::Dirichlet; 6],
            max_modes: [1, 1, 1],
        };

        let solution = RectangularSolution::new(params);

        // At the center of the domain, sin(x/2) = sin(π/2) = 1 for all dimensions
        let value = solution.evaluate_at(PI / 2.0, PI / 2.0, PI / 2.0);
        assert_abs_diff_eq!(value.norm(), 1.0, epsilon = 1e-10);

        // At the boundary, the solution should be zero
        let value = solution.evaluate_at(0.0, PI / 2.0, PI / 2.0);
        assert_abs_diff_eq!(value.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bessel_function() {
        // Test known values
        assert_abs_diff_eq!(bessel_j(0, 0.0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bessel_j(1, 0.0), 0.0, epsilon = 1e-10);

        // Test J_0(1) ≈ 0.7652
        assert_abs_diff_eq!(bessel_j(0, 1.0), 0.7651976866, epsilon = 1e-8);
    }

    #[test]
    fn test_spherical_bessel_function() {
        // Test known values
        assert_abs_diff_eq!(spherical_bessel_j(0, 0.0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(spherical_bessel_j(1, 0.0), 0.0, epsilon = 1e-10);

        // Test j_0(π) = sin(π)/π = 0
        assert_abs_diff_eq!(spherical_bessel_j(0, PI), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solution_comparison() {
        let shape = (4, 4, 4);
        let numerical = WaveArray::from_scalar(shape, Complex64::new(1.0, 0.1));
        let analytical = WaveArray::from_scalar(shape, Complex64::new(1.0, 0.0));

        let (l2_error, max_error, rel_error) = compare_solutions(&numerical, &analytical);

        assert!(l2_error > 0.0);
        assert!(max_error > 0.0);
        assert!(rel_error > 0.0);
    }
}
