//! Schwarz domain decomposition method for Helmholtz equation
//!
//! This module implements an overlapping Schwarz method for parallel
//! solution of the Helmholtz equation, which avoids global FFT operations.

use crate::domain::helmholtz::HelmholtzDomain;
use crate::domain_decomposition::{DomainDecomposition, Subdomain};
use crate::engine::array::{Complex64, WaveArray};
use crate::engine::operations::mix;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

/// Schwarz domain decomposition solver for Helmholtz equation
#[derive(Debug)]
pub struct SchwarzHelmholtzSolver {
    /// Global domain parameters
    global_domain: Arc<HelmholtzDomain>,
    /// Domain decomposition
    _decomposition: Arc<DomainDecomposition>,
    /// Local subdomains with their own Helmholtz solvers
    local_domains: Vec<LocalHelmholtzDomain>,
    /// Overlap size for Schwarz method (typically 4-8 grid points)
    overlap: usize,
    /// Relaxation parameter for Schwarz iteration
    theta: f64,
}

/// Local subdomain with its own Helmholtz solver
#[derive(Debug)]
struct LocalHelmholtzDomain {
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

impl SchwarzHelmholtzSolver {
    /// Create a new Schwarz domain decomposition solver
    pub fn new(
        global_domain: HelmholtzDomain,
        num_subdomains: (usize, usize, usize),
        overlap: usize,
    ) -> Self {
        let shape = global_domain.shape;
        let decomposition = Arc::new(DomainDecomposition::new(shape, num_subdomains, overlap));

        // Create local domains
        let local_domains = Self::create_local_domains(&global_domain, &decomposition, overlap);

        Self {
            global_domain: Arc::new(global_domain),
            _decomposition: decomposition,
            local_domains,
            overlap,
            theta: 0.6, // Relaxation parameter (typically 0.5-0.7)
        }
    }

    /// Create local subdomain solvers
    fn create_local_domains(
        global_domain: &HelmholtzDomain,
        decomposition: &DomainDecomposition,
        overlap: usize,
    ) -> Vec<LocalHelmholtzDomain> {
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

                // Create local solver with absorbing boundaries at subdomain edges
                let local_solver = HelmholtzDomain::new(
                    local_perm,
                    global_domain.pixel_size,
                    global_domain.wavelength,
                    [false, false, false],    // Non-periodic for subdomains
                    [[0, 0], [0, 0], [0, 0]], // No PML - use Schwarz overlap instead
                );

                LocalHelmholtzDomain {
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
        // DEBUG: Check global source magnitude
        let global_max = global_source
            .data
            .iter()
            .map(|c: &Complex64| c.norm())
            .fold(0.0, f64::max);
        println!("DEBUG: Global source max magnitude: {:.2e}", global_max);
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

    /// Perform one Schwarz iteration
    pub fn schwarz_iteration(&self) -> f64 {
        // Parallel solve on each subdomain
        let local_residuals: Vec<f64> = self
            .local_domains
            .par_iter()
            .map(|local| {
                // Get boundary conditions from neighbors
                let bc = self.get_boundary_conditions(local);

                // Solve local problem with boundary conditions
                let residual = self.solve_local_with_bc(local, &bc);

                residual
            })
            .collect();

        // Return maximum residual
        local_residuals.into_iter().fold(0.0, f64::max)
    }

    /// Get boundary conditions from neighboring subdomains
    fn get_boundary_conditions(&self, local: &LocalHelmholtzDomain) -> WaveArray<Complex64> {
        let local_shape = local.solution.read().unwrap().shape_tuple();
        let mut bc = WaveArray::zeros(local_shape);

        // Copy data from neighbors at overlap regions
        for other in &self.local_domains {
            if other.subdomain.id == local.subdomain.id {
                continue;
            }

            // Check for overlap and copy data
            if Self::subdomains_overlap(local, other) {
                Self::copy_overlap_data(local, other, &mut bc);
            }
        }

        bc
    }

    /// Check if two subdomains overlap
    fn subdomains_overlap(a: &LocalHelmholtzDomain, b: &LocalHelmholtzDomain) -> bool {
        // Check if extended regions overlap
        for dim in 0..3 {
            if a.extended_end[dim] <= b.extended_start[dim]
                || b.extended_end[dim] <= a.extended_start[dim]
            {
                return false;
            }
        }
        true
    }

    /// Copy overlap data between subdomains
    fn copy_overlap_data(
        target: &LocalHelmholtzDomain,
        source: &LocalHelmholtzDomain,
        bc: &mut WaveArray<Complex64>,
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

        // Copy data
        for gi in overlap_start[0]..overlap_end[0] {
            for gj in overlap_start[1]..overlap_end[1] {
                for gk in overlap_start[2]..overlap_end[2] {
                    // Convert to local indices
                    let ti = gi - target.extended_start[0];
                    let tj = gj - target.extended_start[1];
                    let tk = gk - target.extended_start[2];

                    let si = gi - source.extended_start[0];
                    let sj = gj - source.extended_start[1];
                    let sk = gk - source.extended_start[2];

                    bc.data[[ti, tj, tk]] = source_solution.data[[si, sj, sk]];
                }
            }
        }
    }

    /// Solve local problem with boundary conditions
    fn solve_local_with_bc(&self, local: &LocalHelmholtzDomain, bc: &WaveArray<Complex64>) -> f64 {
        use crate::domain::iteration::{preconditioned_richardson, IterationConfig};

        let mut solution = local.solution.write().unwrap();
        let old_solution = solution.clone();

        // Initialize solution with boundary conditions in overlap regions
        // This is the key: BC provides initial guess from neighbors
        for i in 0..solution.shape_tuple().0 {
            for j in 0..solution.shape_tuple().1 {
                for k in 0..solution.shape_tuple().2 {
                    if bc.data[[i, j, k]].norm() > 1e-10 {
                        // Use BC value as initial guess in overlap region
                        solution.data[[i, j, k]] = bc.data[[i, j, k]];
                    }
                }
            }
        }

        // Solve local Helmholtz problem: (I - P*M)*u = P*source
        // Use preconditioned Richardson with more iterations
        let iter_config = IterationConfig {
            max_iterations: 500, // More iterations per subdomain
            threshold: 1e-6,     // Tighter tolerance
            alpha: 0.75,
            full_residuals: false,
        };

        let result = preconditioned_richardson(&local.solver, &local.source, iter_config);

        // Blend result with boundary conditions using relaxation
        // This implements additive Schwarz with relaxation
        for i in 0..solution.shape_tuple().0 {
            for j in 0..solution.shape_tuple().1 {
                for k in 0..solution.shape_tuple().2 {
                    if bc.data[[i, j, k]].norm() > 1e-10 {
                        // In overlap region: blend between new solution and BC
                        solution.data[[i, j, k]] = result.field.data[[i, j, k]] * self.theta
                            + bc.data[[i, j, k]] * (1.0 - self.theta);
                    } else {
                        // In interior: use computed solution
                        solution.data[[i, j, k]] = result.field.data[[i, j, k]];
                    }
                }
            }
        }

        // Compute change in solution (convergence measure)
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
    /// Compute smooth weight function for partition of unity
    /// Uses a cosine-based transition function in overlap regions
    fn compute_smooth_weight(
        gi: usize,
        gj: usize,
        gk: usize,
        subdomain_start: [usize; 3],
        subdomain_end: [usize; 3],
        _extended_start: [usize; 3],
        _extended_end: [usize; 3],
        overlap: usize,
    ) -> f64 {
        use std::f64::consts::PI;

        // Calculate distance to subdomain boundaries
        let mut weight = 1.0;

        // For each dimension, compute transition weight if in overlap region
        for dim in 0..3 {
            let pos = [gi, gj, gk][dim];
            let start = subdomain_start[dim];
            let end = subdomain_end[dim];

            if pos < start {
                // In left overlap region
                let dist_from_boundary = (start - pos) as f64;
                if dist_from_boundary < overlap as f64 {
                    // Smooth transition from 0 to 1 using cosine
                    let t = dist_from_boundary / overlap as f64;
                    weight *= 0.5 * (1.0 - (PI * t).cos());
                } else {
                    weight = 0.0;
                    break;
                }
            } else if pos >= end {
                // In right overlap region
                let dist_from_boundary = (pos - end + 1) as f64;
                if dist_from_boundary < overlap as f64 {
                    // Smooth transition from 1 to 0 using cosine
                    let t = dist_from_boundary / overlap as f64;
                    weight *= 0.5 * (1.0 + (PI * t).cos());
                } else {
                    weight = 0.0;
                    break;
                }
            }
            // else: pos is in [start, end), use full weight (1.0)
        }

        weight
    }

    /// Gather global solution from local subdomains
    pub fn gather_solution(&self) -> WaveArray<Complex64> {
        let mut global = WaveArray::zeros(self.global_domain.shape);
        let mut weights =
            vec![
                vec![vec![0.0; self.global_domain.shape.2]; self.global_domain.shape.1];
                self.global_domain.shape.0
            ];

        // Accumulate solutions with proper partition of unity
        for local in &self.local_domains {
            let local_solution = local.solution.read().unwrap();

            // Calculate the actual subdomain boundaries (without overlap)
            let subdomain_start = local.subdomain.start;
            let subdomain_end = local.subdomain.end;

            // Copy all points from the extended domain with smooth weighting
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

                        // Compute smooth weight function for partition of unity
                        let weight = Self::compute_smooth_weight(
                            gi,
                            gj,
                            gk,
                            subdomain_start,
                            subdomain_end,
                            local.extended_start,
                            local.extended_end,
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

        // Normalize by weights to ensure partition of unity
        for i in 0..self.global_domain.shape.0 {
            for j in 0..self.global_domain.shape.1 {
                for k in 0..self.global_domain.shape.2 {
                    if weights[i][j][k] > 1e-10 {
                        global.data[[i, j, k]] /= weights[i][j][k];
                    }
                }
            }
        }

        // DEBUG: Check solution magnitude
        let solution_max = global
            .data
            .iter()
            .map(|c: &Complex64| c.norm())
            .fold(0.0, f64::max);
        println!(
            "DEBUG: Gathered solution max magnitude: {:.2e}",
            solution_max
        );

        // DEBUG: Check for zeros in the solution
        let num_zeros = global.data.iter().filter(|c| c.norm() < 1e-15).count();
        let total_points =
            self.global_domain.shape.0 * self.global_domain.shape.1 * self.global_domain.shape.2;
        if num_zeros > 0 {
            println!(
                "DEBUG: Warning - {} out of {} points are zero in gathered solution",
                num_zeros, total_points
            );
        }

        global
    }
}

/// Solve Helmholtz equation using Schwarz domain decomposition
pub fn solve_helmholtz_schwarz(
    global_domain: HelmholtzDomain,
    source: WaveArray<Complex64>,
    num_subdomains: (usize, usize, usize),
    max_iterations: usize,
    tolerance: f64,
) -> WaveArray<Complex64> {
    println!(
        "Solving Helmholtz equation using Schwarz method with {} subdomains...",
        num_subdomains.0 * num_subdomains.1 * num_subdomains.2
    );

    // Create Schwarz solver
    let mut solver = SchwarzHelmholtzSolver::new(
        global_domain,
        num_subdomains,
        4, // Overlap of 4 grid points
    );

    // Set source
    solver.set_source(&source);

    // Iterate
    for iter in 0..max_iterations {
        let residual = solver.schwarz_iteration();

        if iter % 10 == 0 {
            println!("Schwarz iteration {}: residual = {:.2e}", iter, residual);
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

    // Gather and return solution
    solver.gather_solution()
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_schwarz_solver_creation() {
        let shape = (32, 32, 32);
        let permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

        let domain = HelmholtzDomain::new(
            permittivity,
            0.1,
            0.5,
            [false, false, false],
            [[2, 2], [2, 2], [2, 2]],
        );

        let solver = SchwarzHelmholtzSolver::new(domain, (2, 2, 2), 4);

        assert_eq!(solver.local_domains.len(), 8);
        assert_eq!(solver.overlap, 4);
    }
}
