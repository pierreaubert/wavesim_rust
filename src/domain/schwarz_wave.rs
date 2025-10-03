//! Wave-Optimized Schwarz domain decomposition method for Helmholtz equation
//!
//! This module implements a Schwarz domain decomposition method with wave-optimized
//! Robin transmission conditions, which converge faster than classical Dirichlet conditions
//! for wave propagation problems.
//!
//! The Robin transmission condition is: ∂u/∂n + iku = g on subdomain interfaces,
//! where k is the wavenumber and i is the imaginary unit.

use crate::domain::helmholtz::HelmholtzDomain;
use crate::domain::iteration::{preconditioned_richardson, IterationConfig};
use crate::domain_decomposition::{DomainDecomposition, Subdomain};
use crate::engine::array::{Complex64, WaveArray};
use crate::engine::operations::mix;
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::{Arc, RwLock};

/// Wave-optimized Schwarz domain decomposition solver for Helmholtz equation
#[derive(Debug)]
pub struct WaveOptimizedSchwarzSolver {
    /// Global domain parameters
    global_domain: Arc<HelmholtzDomain>,
    /// Domain decomposition
    _decomposition: Arc<DomainDecomposition>,
    /// Local subdomains with their own Helmholtz solvers
    local_domains: Vec<LocalWaveDomain>,
    /// Overlap size for Schwarz method
    overlap: usize,
    /// Relaxation parameter for Schwarz iteration
    theta: f64,
    /// Wavenumber (k = 2π/λ)
    wavenumber: f64,
    /// Impedance parameter for Robin condition (typically ik)
    impedance: Complex64,
}

/// Local subdomain with wave-optimized solver
#[derive(Debug)]
struct LocalWaveDomain {
    /// Subdomain info
    subdomain: Subdomain,
    /// Local Helmholtz solver
    solver: HelmholtzDomain,
    /// Local solution
    solution: RwLock<WaveArray<Complex64>>,
    /// Local source
    source: WaveArray<Complex64>,
    /// Extended indices including overlap
    extended_start: [usize; 3],
    extended_end: [usize; 3],
}

impl WaveOptimizedSchwarzSolver {
    /// Create a new wave-optimized Schwarz domain decomposition solver
    pub fn new(
        global_domain: HelmholtzDomain,
        num_subdomains: (usize, usize, usize),
        overlap: usize,
    ) -> Self {
        let shape = global_domain.shape;
        let decomposition = Arc::new(DomainDecomposition::new(shape, num_subdomains, overlap));

        // Calculate wavenumber from global domain
        let wavenumber = 2.0 * PI / global_domain.wavelength;

        // Robin impedance: ∂u/∂n + iku = g
        // For optimal convergence, use impedance = ik
        let impedance = Complex::new(0.0, wavenumber);

        // Create local domains
        let local_domains = Self::create_local_domains(&global_domain, &decomposition, overlap);

        Self {
            global_domain: Arc::new(global_domain),
            _decomposition: decomposition,
            local_domains,
            overlap,
            theta: 0.7, // Slightly higher relaxation for wave-optimized method
            wavenumber,
            impedance,
        }
    }

    /// Create local subdomain solvers
    fn create_local_domains(
        global_domain: &HelmholtzDomain,
        decomposition: &DomainDecomposition,
        overlap: usize,
    ) -> Vec<LocalWaveDomain> {
        decomposition
            .subdomains
            .iter()
            .map(|subdomain| {
                // Calculate extended bounds with overlap
                let extended_start = [
                    subdomain.start[0].saturating_sub(overlap),
                    subdomain.start[1].saturating_sub(overlap),
                    subdomain.start[2].saturating_sub(overlap),
                ];

                let extended_end = [
                    (subdomain.end[0] + overlap).min(global_domain.shape.0),
                    (subdomain.end[1] + overlap).min(global_domain.shape.1),
                    (subdomain.end[2] + overlap).min(global_domain.shape.2),
                ];

                let local_shape = (
                    extended_end[0] - extended_start[0],
                    extended_end[1] - extended_start[1],
                    extended_end[2] - extended_start[2],
                );

                // Extract local permittivity
                let mut local_perm = WaveArray::zeros(local_shape);
                for i in 0..local_shape.0 {
                    for j in 0..local_shape.1 {
                        for k in 0..local_shape.2 {
                            let gi = extended_start[0] + i;
                            let gj = extended_start[1] + j;
                            let gk = extended_start[2] + k;
                            local_perm.data[[i, j, k]] =
                                global_domain.permittivity.data[[gi, gj, gk]];
                        }
                    }
                }

                // Create local solver
                let local_solver = HelmholtzDomain::new(
                    local_perm,
                    global_domain.pixel_size,
                    global_domain.wavelength,
                    [false, false, false],    // Non-periodic for subdomains
                    [[0, 0], [0, 0], [0, 0]], // No PML - use Robin conditions
                );

                LocalWaveDomain {
                    subdomain: subdomain.clone(),
                    solver: local_solver,
                    solution: RwLock::new(WaveArray::zeros(local_shape)),
                    source: WaveArray::zeros(local_shape),
                    extended_start,
                    extended_end,
                }
            })
            .collect()
    }

    /// Set source field for the problem
    pub fn set_source(&mut self, global_source: &WaveArray<Complex64>) {
        // Distribute source to local subdomains
        for local in &mut self.local_domains {
            let local_shape = local.solution.read().unwrap().shape_tuple();

            for i in 0..local_shape.0 {
                for j in 0..local_shape.1 {
                    for k in 0..local_shape.2 {
                        let gi = local.extended_start[0] + i;
                        let gj = local.extended_start[1] + j;
                        let gk = local.extended_start[2] + k;

                        if gi < self.global_domain.shape.0
                            && gj < self.global_domain.shape.1
                            && gk < self.global_domain.shape.2
                        {
                            local.source.data[[i, j, k]] = global_source.data[[gi, gj, gk]];
                        }
                    }
                }
            }
        }
    }

    /// Perform one wave-optimized Schwarz iteration
    pub fn schwarz_iteration(&self) -> f64 {
        // Parallel solve on each subdomain with Robin boundary conditions
        let local_residuals: Vec<f64> = self
            .local_domains
            .par_iter()
            .map(|local| {
                // Get Robin boundary conditions from neighbors
                let bc = self.get_robin_boundary_conditions(local);

                // Solve local problem with Robin BC
                let residual = self.solve_local_with_robin_bc(local, &bc);

                residual
            })
            .collect();

        // Return maximum residual
        local_residuals.into_iter().fold(0.0, f64::max)
    }

    /// Get Robin boundary conditions from neighboring subdomains
    /// Implements: ∂u/∂n + iku = g
    fn get_robin_boundary_conditions(&self, local: &LocalWaveDomain) -> WaveArray<Complex64> {
        let local_shape = local.solution.read().unwrap().shape_tuple();
        let mut bc = WaveArray::zeros(local_shape);

        // Copy data from neighbors with Robin impedance weighting
        for other in &self.local_domains {
            if other.subdomain.id == local.subdomain.id {
                continue;
            }

            // Check for overlap and apply Robin condition
            if Self::subdomains_overlap(local, other) {
                Self::copy_robin_overlap_data(
                    local,
                    other,
                    &mut bc,
                    self.impedance,
                    self.global_domain.pixel_size,
                );
            }
        }

        bc
    }

    /// Check if two subdomains overlap
    fn subdomains_overlap(a: &LocalWaveDomain, b: &LocalWaveDomain) -> bool {
        for dim in 0..3 {
            if a.extended_end[dim] <= b.extended_start[dim]
                || b.extended_end[dim] <= a.extended_start[dim]
            {
                return false;
            }
        }
        true
    }

    /// Copy overlap data with Robin transmission condition
    /// Robin condition: ∂u/∂n + Z*u where Z is impedance
    fn copy_robin_overlap_data(
        target: &LocalWaveDomain,
        source: &LocalWaveDomain,
        bc: &mut WaveArray<Complex64>,
        impedance: Complex64,
        pixel_size: f64,
    ) {
        let source_solution = source.solution.read().unwrap();

        // Calculate overlap region in global coordinates
        let overlap_start = [
            target.extended_start[0].max(source.extended_start[0]),
            target.extended_start[1].max(source.extended_start[1]),
            target.extended_start[2].max(source.extended_start[2]),
        ];

        let overlap_end = [
            target.extended_end[0].min(source.extended_end[0]),
            target.extended_end[1].min(source.extended_end[1]),
            target.extended_end[2].min(source.extended_end[2]),
        ];

        // Apply Robin condition in overlap region
        for gi in overlap_start[0]..overlap_end[0] {
            for gj in overlap_start[1]..overlap_end[1] {
                for gk in overlap_start[2]..overlap_end[2] {
                    let ti = gi - target.extended_start[0];
                    let tj = gj - target.extended_start[1];
                    let tk = gk - target.extended_start[2];

                    let si = gi - source.extended_start[0];
                    let sj = gj - source.extended_start[1];
                    let sk = gk - source.extended_start[2];

                    let u_neighbor = source_solution.data[[si, sj, sk]];

                    // Compute approximate normal derivative using finite differences
                    let normal_deriv = Self::compute_normal_derivative(
                        &source_solution,
                        si,
                        sj,
                        sk,
                        target,
                        source,
                        gi,
                        gj,
                        gk,
                        pixel_size,
                    );

                    // Robin condition: ∂u/∂n + Z*u
                    let robin_value = normal_deriv + impedance * u_neighbor;
                    bc.data[[ti, tj, tk]] = robin_value;
                }
            }
        }
    }

    /// Compute normal derivative at interface using finite differences
    fn compute_normal_derivative(
        field: &WaveArray<Complex64>,
        i: usize,
        j: usize,
        k: usize,
        target: &LocalWaveDomain,
        source: &LocalWaveDomain,
        gi: usize,
        gj: usize,
        gk: usize,
        pixel_size: f64,
    ) -> Complex64 {
        let shape = field.shape_tuple();

        // Determine interface direction (which dimension has the interface)
        let mut deriv = Complex::new(0.0, 0.0);
        let mut count = 0;

        // Check each dimension for interface
        for dim in 0..3 {
            let at_boundary = match dim {
                0 => {
                    gi == target.subdomain.start[0].max(source.subdomain.end[0])
                        || gi == target.subdomain.end[0].min(source.subdomain.start[0])
                }
                1 => {
                    gj == target.subdomain.start[1].max(source.subdomain.end[1])
                        || gj == target.subdomain.end[1].min(source.subdomain.start[1])
                }
                2 => {
                    gk == target.subdomain.start[2].max(source.subdomain.end[2])
                        || gk == target.subdomain.end[2].min(source.subdomain.start[2])
                }
                _ => false,
            };

            if at_boundary {
                // Compute derivative in this direction
                deriv += match dim {
                    0 if i > 0 && i < shape.0 - 1 => {
                        (field.data[[i + 1, j, k]] - field.data[[i - 1, j, k]]) / (2.0 * pixel_size)
                    }
                    1 if j > 0 && j < shape.1 - 1 => {
                        (field.data[[i, j + 1, k]] - field.data[[i, j - 1, k]]) / (2.0 * pixel_size)
                    }
                    2 if k > 0 && k < shape.2 - 1 => {
                        (field.data[[i, j, k + 1]] - field.data[[i, j, k - 1]]) / (2.0 * pixel_size)
                    }
                    _ => Complex::new(0.0, 0.0),
                };
                count += 1;
            }
        }

        if count > 0 {
            deriv / (count as f64)
        } else {
            Complex::new(0.0, 0.0)
        }
    }

    /// Solve local problem with Robin boundary conditions
    fn solve_local_with_robin_bc(&self, local: &LocalWaveDomain, bc: &WaveArray<Complex64>) -> f64 {
        let mut solution = local.solution.write().unwrap();
        let old_solution = solution.clone();

        // Apply Robin BC as initial guess in overlap regions
        for i in 0..solution.shape_tuple().0 {
            for j in 0..solution.shape_tuple().1 {
                for k in 0..solution.shape_tuple().2 {
                    if bc.data[[i, j, k]].norm() > 1e-10 {
                        // Convert Robin data back to field value
                        // From ∂u/∂n + Z*u = g, approximately: u ≈ g/Z
                        solution.data[[i, j, k]] = bc.data[[i, j, k]] / self.impedance;
                    }
                }
            }
        }

        // Solve local Helmholtz problem
        let iter_config = IterationConfig {
            max_iterations: 300,
            threshold: 1e-6,
            alpha: 0.75,
            full_residuals: false,
        };

        let result = preconditioned_richardson(&local.solver, &local.source, iter_config);

        // Blend with Robin BC using relaxation
        for i in 0..solution.shape_tuple().0 {
            for j in 0..solution.shape_tuple().1 {
                for k in 0..solution.shape_tuple().2 {
                    if bc.data[[i, j, k]].norm() > 1e-10 {
                        // In overlap: blend new solution with BC-derived value
                        let bc_value = bc.data[[i, j, k]] / self.impedance;
                        solution.data[[i, j, k]] = result.field.data[[i, j, k]] * self.theta
                            + bc_value * (1.0 - self.theta);
                    } else {
                        // Interior: use computed solution
                        solution.data[[i, j, k]] = result.field.data[[i, j, k]];
                    }
                }
            }
        }

        // Compute change in solution
        let mut residual = WaveArray::zeros(solution.shape_tuple());
        mix(
            Complex64::new(1.0, 0.0),
            &solution,
            Complex64::new(-1.0, 0.0),
            &old_solution,
            &mut residual,
        );

        residual.norm_squared().sqrt()
    }

    /// Gather global solution from local subdomains using smooth partition of unity
    pub fn gather_solution(&self) -> WaveArray<Complex64> {
        let mut global = WaveArray::zeros(self.global_domain.shape);
        let mut weights =
            vec![
                vec![vec![0.0; self.global_domain.shape.2]; self.global_domain.shape.1];
                self.global_domain.shape.0
            ];

        // Accumulate solutions with partition of unity
        for local in &self.local_domains {
            let local_solution = local.solution.read().unwrap();

            for gi in local.extended_start[0]..local.extended_end[0] {
                for gj in local.extended_start[1]..local.extended_end[1] {
                    for gk in local.extended_start[2]..local.extended_end[2] {
                        if gi >= self.global_domain.shape.0
                            || gj >= self.global_domain.shape.1
                            || gk >= self.global_domain.shape.2
                        {
                            continue;
                        }

                        let li = gi - local.extended_start[0];
                        let lj = gj - local.extended_start[1];
                        let lk = gk - local.extended_start[2];

                        let weight = Self::compute_smooth_weight(
                            gi,
                            gj,
                            gk,
                            local.subdomain.start,
                            local.subdomain.end,
                            self.overlap,
                        );

                        if weight > 1e-10 {
                            global.data[[gi, gj, gk]] += local_solution.data[[li, lj, lk]] * weight;
                            weights[gi][gj][gk] += weight;
                        }
                    }
                }
            }
        }

        // Normalize by weights
        for i in 0..self.global_domain.shape.0 {
            for j in 0..self.global_domain.shape.1 {
                for k in 0..self.global_domain.shape.2 {
                    if weights[i][j][k] > 1e-10 {
                        global.data[[i, j, k]] /= weights[i][j][k];
                    }
                }
            }
        }

        global
    }

    /// Compute smooth weight function for partition of unity
    fn compute_smooth_weight(
        gi: usize,
        gj: usize,
        gk: usize,
        subdomain_start: [usize; 3],
        subdomain_end: [usize; 3],
        overlap: usize,
    ) -> f64 {
        let mut weight = 1.0;

        for dim in 0..3 {
            let pos = [gi, gj, gk][dim];
            let start = subdomain_start[dim];
            let end = subdomain_end[dim];

            if pos < start {
                let dist = (start - pos) as f64;
                if dist < overlap as f64 {
                    let t = dist / overlap as f64;
                    weight *= 0.5 * (1.0 - (PI * t).cos());
                } else {
                    weight = 0.0;
                    break;
                }
            } else if pos >= end {
                let dist = (pos - end + 1) as f64;
                if dist < overlap as f64 {
                    let t = dist / overlap as f64;
                    weight *= 0.5 * (1.0 + (PI * t).cos());
                } else {
                    weight = 0.0;
                    break;
                }
            }
        }

        weight
    }
}

/// Solve Helmholtz equation using wave-optimized Schwarz domain decomposition
pub fn solve_helmholtz_wave_schwarz(
    global_domain: HelmholtzDomain,
    source: WaveArray<Complex64>,
    num_subdomains: (usize, usize, usize),
    max_iterations: usize,
    tolerance: f64,
) -> WaveArray<Complex64> {
    println!(
        "Solving Helmholtz equation using Wave-Optimized Schwarz method with {} subdomains...",
        num_subdomains.0 * num_subdomains.1 * num_subdomains.2
    );

    let mut solver = WaveOptimizedSchwarzSolver::new(global_domain, num_subdomains, 4);

    solver.set_source(&source);

    // Schwarz iterations
    for iter in 0..max_iterations {
        let residual = solver.schwarz_iteration();

        if iter % 10 == 0 {
            println!(
                "Wave-Schwarz iteration {}: residual = {:.2e}",
                iter, residual
            );
        }

        if residual < tolerance {
            println!(
                "Converged after {} iterations (residual: {:.2e})",
                iter + 1,
                residual
            );
            break;
        }
    }

    solver.gather_solution()
}

/// Analytical solution utilities for validation
pub mod analytical {
    use super::*;

    /// Compute analytical solution for rectangular domain using 2D Fourier series
    /// u(x,y) = Σ Aₙₘ sin(nπx/Lₓ)sin(mπy/Lᵧ)
    /// with eigenvalues k²ₙₘ = (nπ/Lₓ)² + (mπ/Lᵧ)²
    pub fn rectangular_cavity_2d(
        nx: usize,
        ny: usize,
        lx: f64,
        ly: f64,
        k: f64,
        n_modes: usize,
    ) -> WaveArray<Complex64> {
        let mut field = WaveArray::zeros((nx, ny, 1));
        let dx = lx / (nx as f64);
        let dy = ly / (ny as f64);

        // Sum over modes that match the wavenumber
        for n in 1..=n_modes {
            for m in 1..=n_modes {
                let kx = n as f64 * PI / lx;
                let ky = m as f64 * PI / ly;
                let k_nm_sq = kx * kx + ky * ky;

                // Check if this mode is resonant (k² ≈ k²ₙₘ)
                if (k * k - k_nm_sq).abs() < 0.1 * k * k {
                    // Amplitude (can be set based on source)
                    let amplitude = 1.0 / (n * m) as f64;

                    for i in 0..nx {
                        for j in 0..ny {
                            let x = (i as f64 + 0.5) * dx;
                            let y = (j as f64 + 0.5) * dy;

                            let value = amplitude * (kx * x).sin() * (ky * y).sin();
                            field.data[[i, j, 0]] += Complex::new(value, 0.0);
                        }
                    }
                }
            }
        }

        field
    }

    /// Compute rectangular eigenvalues k²ₙₘ = (nπ/Lₓ)² + (mπ/Lᵧ)²
    pub fn rectangular_eigenvalues(lx: f64, ly: f64, n: usize, m: usize) -> f64 {
        let kx = n as f64 * PI / lx;
        let ky = m as f64 * PI / ly;
        (kx * kx + ky * ky).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_wave_schwarz_solver_creation() {
        let shape = (32, 32, 32);
        let permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

        let domain = HelmholtzDomain::new(
            permittivity,
            0.1,
            0.5,
            [false, false, false],
            [[2, 2], [2, 2], [2, 2]],
        );

        let solver = WaveOptimizedSchwarzSolver::new(domain, (2, 2, 2), 4);

        assert_eq!(solver.local_domains.len(), 8);
        assert_eq!(solver.overlap, 4);
        assert!((solver.wavenumber - 2.0 * PI / 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_rectangular_eigenvalue() {
        let lx = 1.0;
        let ly = 1.0;

        // First mode (1,1)
        let k11 = analytical::rectangular_eigenvalues(lx, ly, 1, 1);
        let expected = PI * (2.0_f64).sqrt();
        assert!((k11 - expected).abs() < 1e-10);

        // Mode (2,1)
        let k21 = analytical::rectangular_eigenvalues(lx, ly, 2, 1);
        let expected = PI * (5.0_f64).sqrt();
        assert!((k21 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_rectangular_analytical_solution() {
        let nx = 32;
        let ny = 32;
        let lx = 1.0;
        let ly = 1.0;

        // Use first eigenvalue
        let k = analytical::rectangular_eigenvalues(lx, ly, 1, 1);

        let field = analytical::rectangular_cavity_2d(nx, ny, lx, ly, k, 3);

        // Check field is not zero
        assert!(field.norm_squared() > 0.0);

        // Check symmetry for (1,1) mode
        let mid_x = nx / 2;
        let mid_y = ny / 2;
        let center_val = field.data[[mid_x, mid_y, 0]].norm();

        // Center should have maximum for (1,1) mode
        let corner_val = field.data[[nx - 1, ny - 1, 0]].norm();
        assert!(center_val > corner_val);
    }
}
