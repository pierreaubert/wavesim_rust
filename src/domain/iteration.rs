//! Iteration methods for solving wave equations
//!
//! Implements the preconditioned Richardson iteration method for the Modified Born Series

use crate::domain::domain_trait::Domain;
use crate::engine::array::{Complex64, WaveArray};
use crate::engine::operations::{mix, scale};

/// Result of an iteration
#[derive(Debug, Clone)]
pub struct IterationResult {
    /// The computed field
    pub field: WaveArray<Complex64>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm
    pub residual_norm: f64,
    /// History of residual norms (if requested)
    pub residual_history: Option<Vec<f64>>,
}

/// Configuration for the preconditioned Richardson iteration
#[derive(Debug, Clone)]
pub struct IterationConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence threshold for residual norm
    pub threshold: f64,
    /// Relaxation parameter (typically 0.75)
    pub alpha: f64,
    /// Whether to record full residual history
    pub full_residuals: bool,
}

impl Default for IterationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100000,
            threshold: 1e-6,
            alpha: 0.75,
            full_residuals: false,
        }
    }
}

/// Preconditioned Richardson iteration solver
///
/// Solves the equation A x = y using the iteration:
/// x_{n+1} = x_n + α Γ^{-1} (y - A x_n)
///
/// where Γ^{-1} is the preconditioner
pub fn preconditioned_richardson<D: Domain>(
    domain: &D,
    source: &WaveArray<Complex64>,
    config: IterationConfig,
) -> IterationResult {
    let shape = domain.shape();

    // Initialize field to zero
    let mut x = WaveArray::zeros(shape);
    let mut tmp = WaveArray::zeros(shape);

    // Compute initial residual norm for normalization
    preconditioner(domain, source, &mut tmp);
    let init_norm = tmp.norm_squared();

    if init_norm < 1e-20 {
        // Source is essentially zero
        return IterationResult {
            field: x,
            iterations: 0,
            residual_norm: 0.0,
            residual_history: if config.full_residuals {
                Some(vec![0.0])
            } else {
                None
            },
        };
    }

    let mut residual_history = if config.full_residuals {
        Some(Vec::with_capacity(config.max_iterations))
    } else {
        None
    };

    let mut iterations = 0;
    let mut residual_norm = f64::INFINITY;

    // Main iteration loop
    for i in 0..config.max_iterations {
        iterations = i + 1;

        // Perform one iteration and compute residual
        residual_norm =
            preconditioned_iteration(domain, &mut x, source, &mut tmp, config.alpha, true)
                / init_norm;

        if let Some(ref mut history) = residual_history {
            history.push(residual_norm);
        }

        // Check convergence
        if residual_norm < config.threshold {
            break;
        }
    }

    IterationResult {
        field: x,
        iterations,
        residual_norm,
        residual_history,
    }
}

/// Perform one preconditioned Richardson iteration
///
/// Updates x in place: x -> x + α Γ^{-1} (y - A x)
/// Returns the squared norm of the residual if requested
fn preconditioned_iteration<D: Domain>(
    domain: &D,
    x: &mut WaveArray<Complex64>,
    source: &WaveArray<Complex64>,
    tmp: &mut WaveArray<Complex64>,
    alpha: f64,
    compute_norm: bool,
) -> f64 {
    // Compute B x (medium operator)
    domain.medium(x, tmp);

    // Add scaled source: tmp = B x + c y
    let c = -domain.scale();
    let tmp_clone = tmp.clone();
    mix(Complex64::new(1.0, 0.0), &tmp_clone, c, source, tmp);

    // Apply propagator: tmp = (L+1)^{-1} (B x + c y)
    let tmp2 = tmp.clone();
    domain.propagator(&tmp2, tmp);

    // Compute residual: tmp = x - (L+1)^{-1} (B x + c y)
    let tmp_clone2 = tmp.clone();
    mix(
        Complex64::new(1.0, 0.0),
        x,
        -Complex64::new(1.0, 0.0),
        &tmp_clone2,
        tmp,
    );

    // Apply medium operator to residual: tmp = B [x - (L+1)^{-1} (B x + c y)]
    let tmp3 = tmp.clone();
    domain.medium(&tmp3, tmp);

    // Compute norm if requested
    let norm = if compute_norm {
        tmp.norm_squared()
    } else {
        0.0
    };

    // Update x: x = x - α B [x - (L+1)^{-1} (B x + c y)]
    let x_clone = x.clone();
    mix(
        Complex64::new(1.0, 0.0),
        &x_clone,
        Complex64::new(-alpha, 0.0),
        tmp,
        x,
    );

    norm
}

/// Apply the forward operator A = c^{-1} (L + V)
pub fn forward<D: Domain>(domain: &D, x: &WaveArray<Complex64>, out: &mut WaveArray<Complex64>) {
    // Apply medium operator (1 - V)
    domain.medium(x, out);

    // Apply inverse propagator (L + 1)
    let mut tmp = WaveArray::zeros(domain.shape());
    domain.inverse_propagator(x, &mut tmp);

    // Combine: c^{-1} (L + V) x
    let c = domain.scale();
    let out_clone = out.clone();
    mix(
        Complex64::new(1.0, 0.0) / c,
        &tmp,
        -Complex64::new(1.0, 0.0) / c,
        &out_clone,
        out,
    );
}

/// Apply the preconditioner Γ^{-1} = c B (L+1)^{-1}
pub fn preconditioner<D: Domain>(
    domain: &D,
    x: &WaveArray<Complex64>,
    out: &mut WaveArray<Complex64>,
) {
    // Apply propagator (L+1)^{-1}
    domain.propagator(x, out);

    // Apply medium operator B
    let tmp = out.clone();
    domain.medium(&tmp, out);

    // Scale by c
    let s = domain.scale();
    let out_clone = out.clone();
    scale(s, &out_clone, None, out);
}

/// Apply the preconditioned operator Γ^{-1} A
pub fn preconditioned_operator<D: Domain>(
    domain: &D,
    x: &WaveArray<Complex64>,
    out: &mut WaveArray<Complex64>,
) {
    // Apply B x
    domain.medium(x, out);

    // Apply (L+1)^{-1} B x
    let tmp = out.clone();
    domain.propagator(&tmp, out);

    // Apply B (L+1)^{-1} B x
    let tmp2 = out.clone();
    domain.medium(&tmp2, out);

    // Compute B x - B (L+1)^{-1} B x
    let mut tmp2 = WaveArray::zeros(domain.shape());
    domain.medium(x, &mut tmp2);
    let out_clone = out.clone();
    mix(
        Complex64::new(1.0, 0.0),
        &tmp2,
        -Complex64::new(1.0, 0.0),
        &out_clone,
        out,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::helmholtz::HelmholtzDomain;
    use num_complex::Complex;

    #[test]
    fn test_richardson_convergence() {
        // Create a simple test case
        let mut permittivity = WaveArray::from_scalar((10, 10, 10), Complex::new(1.0, 0.0));

        // Add a small perturbation
        permittivity.data[[5, 5, 5]] = Complex::new(1.1, 0.0);

        let domain = HelmholtzDomain::new(
            permittivity,
            0.1, // pixel_size
            0.5, // wavelength
            [false, false, false],
            [[0, 0], [0, 0], [0, 0]],
        );

        // Create a point source
        let mut source = WaveArray::zeros((10, 10, 10));
        source.data[[5, 5, 5]] = Complex::new(1.0, 0.0);

        let config = IterationConfig {
            max_iterations: 100,
            threshold: 1e-4,
            alpha: 0.75,
            full_residuals: true,
        };

        let result = preconditioned_richardson(&domain, &source, config);

        // Check that we converged
        assert!(result.residual_norm < 1e-4);
        assert!(result.iterations > 0);
        assert!(result.iterations < 100);

        // Check that residuals are decreasing
        if let Some(history) = result.residual_history {
            for i in 1..history.len() {
                // Allow for some fluctuation but overall trend should be down
                if i > 5 {
                    assert!(history[i] < history[0]);
                }
            }
        }
    }

    #[test]
    fn test_preconditioner() {
        let permittivity = WaveArray::from_scalar((5, 5, 5), Complex::new(1.0, 0.0));

        let domain = HelmholtzDomain::new(
            permittivity,
            0.1,
            0.5,
            [true, true, true], // periodic
            [[0, 0], [0, 0], [0, 0]],
        );

        let x = WaveArray::from_scalar((5, 5, 5), Complex::new(1.0, 0.0));
        let mut out = WaveArray::zeros((5, 5, 5));

        preconditioner(&domain, &x, &mut out);

        // Check that output is non-zero
        assert!(out.norm_squared() > 0.0);
    }
}
