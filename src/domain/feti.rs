//! FETI (Finite Element Tearing and Interconnecting) method for acoustic Helmholtz equation
//!
//! This module implements the FETI domain decomposition method, which uses a dual formulation
//! with Lagrange multipliers to enforce continuity at subdomain interfaces. Unlike primal methods
//! (like Schwarz), FETI solves for interface unknowns first, then recovers subdomain solutions.
//!
//! The method:
//! 1. Decomposes the domain into non-overlapping subdomains
//! 2. Introduces Lagrange multipliers λ on interfaces for continuity
//! 3. Solves the dual interface problem: F λ = d
//! 4. Computes local solutions using the interface forces

use crate::domain::helmholtz::HelmholtzDomain;
use crate::domain::iteration::{preconditioned_richardson, IterationConfig};
use crate::domain_decomposition::{DomainDecomposition, Subdomain};
use crate::engine::array::{Complex64, WaveArray};
use crate::engine::operations::mix;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

/// FETI acoustic solver for Helmholtz equation
#[derive(Debug)]
pub struct FETIAcousticSolver {
    /// Global domain parameters
    global_domain: Arc<HelmholtzDomain>,
    /// Domain decomposition (non-overlapping)
    _decomposition: Arc<DomainDecomposition>,
    /// Local subdomains
    local_domains: Vec<FETISubdomain>,
    /// Interface information
    interfaces: Vec<Interface>,
    /// Lagrange multipliers on interfaces
    lagrange_multipliers: Arc<Mutex<Vec<Complex64>>>,
    /// Dual operator scaling (for preconditioning)
    dual_scaling: f64,
}

/// Local subdomain for FETI
#[derive(Debug)]
struct FETISubdomain {
    /// Subdomain info
    subdomain: Subdomain,
    /// Local Helmholtz solver
    solver: HelmholtzDomain,
    /// Local solution
    solution: Mutex<WaveArray<Complex64>>,
    /// Local source
    source: WaveArray<Complex64>,
    /// Interface DOF indices for this subdomain
    interface_dofs: Vec<InterfaceDOF>,
}

/// Interface between two subdomains
#[derive(Debug, Clone)]
struct Interface {
    /// ID of the interface
    id: usize,
    /// First subdomain ID
    subdomain_a: usize,
    /// Second subdomain ID
    subdomain_b: usize,
    /// Interface dimension (0=x, 1=y, 2=z)
    dimension: usize,
    /// Global coordinate of the interface
    position: usize,
    /// Number of DOFs on this interface
    num_dofs: usize,
}

/// Degree of freedom on an interface
#[derive(Debug, Clone)]
struct InterfaceDOF {
    /// Interface ID
    interface_id: usize,
    /// Local DOF index within the interface
    local_dof: usize,
    /// Local position within subdomain
    local_pos: [usize; 3],
    /// Sign (+1 or -1) for jump operator
    sign: f64,
}

impl FETIAcousticSolver {
    /// Create a new FETI solver
    pub fn new(
        global_domain: HelmholtzDomain,
        num_subdomains: (usize, usize, usize),
    ) -> Self {
        let shape = global_domain.shape;
        let decomposition = Arc::new(DomainDecomposition::new(shape, num_subdomains, 0));

        // Create local subdomains (non-overlapping)
        let local_domains = Self::create_local_domains(&global_domain, &decomposition);

        // Identify interfaces between subdomains
        let interfaces = Self::create_interfaces(&decomposition);

        // Initialize Lagrange multipliers (one per interface DOF)
        let total_interface_dofs: usize = interfaces.iter().map(|i| i.num_dofs).sum();
        let lagrange_multipliers = Arc::new(Mutex::new(vec![
            Complex::new(0.0, 0.0);
            total_interface_dofs
        ]));

        Self {
            global_domain: Arc::new(global_domain),
            _decomposition: decomposition,
            local_domains,
            interfaces,
            lagrange_multipliers,
            dual_scaling: 1.0,
        }
    }

    /// Create non-overlapping local subdomains
    fn create_local_domains(
        global_domain: &HelmholtzDomain,
        decomposition: &DomainDecomposition,
    ) -> Vec<FETISubdomain> {
        decomposition
            .subdomains
            .iter()
            .map(|subdomain| {
                let local_shape = (
                    subdomain.end[0] - subdomain.start[0],
                    subdomain.end[1] - subdomain.start[1],
                    subdomain.end[2] - subdomain.start[2],
                );

                // Extract local permittivity
                let mut local_perm = WaveArray::zeros(local_shape);
                for i in 0..local_shape.0 {
                    for j in 0..local_shape.1 {
                        for k in 0..local_shape.2 {
                            let gi = subdomain.start[0] + i;
                            let gj = subdomain.start[1] + j;
                            let gk = subdomain.start[2] + k;
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
                    [false, false, false],
                    [[0, 0], [0, 0], [0, 0]],
                );

                FETISubdomain {
                    subdomain: subdomain.clone(),
                    solver: local_solver,
                    solution: Mutex::new(WaveArray::zeros(local_shape)),
                    source: WaveArray::zeros(local_shape),
                    interface_dofs: Vec::new(),
                }
            })
            .collect()
    }

    /// Create interfaces between subdomains
    fn create_interfaces(decomposition: &DomainDecomposition) -> Vec<Interface> {
        let mut interfaces = Vec::new();
        let mut interface_id = 0;

        // Iterate over all subdomain pairs to find interfaces
        for i in 0..decomposition.subdomains.len() {
            let sub_a = &decomposition.subdomains[i];

            // Check neighbors
            for (dim, neighbor_opt) in [
                (0, sub_a.neighbors.right),
                (1, sub_a.neighbors.back),
                (2, sub_a.neighbors.top),
            ]
            .iter()
            {
                if let Some(neighbor_id) = neighbor_opt {
                    if *neighbor_id > i {
                        // Only create interface once
                        let sub_b = &decomposition.subdomains[*neighbor_id];

                        // Determine interface position and size
                        let position = sub_a.end[*dim];
                        let num_dofs = Self::compute_interface_dofs(sub_a, sub_b, *dim);

                        interfaces.push(Interface {
                            id: interface_id,
                            subdomain_a: i,
                            subdomain_b: *neighbor_id,
                            dimension: *dim,
                            position,
                            num_dofs,
                        });

                        interface_id += 1;
                    }
                }
            }
        }

        interfaces
    }

    /// Compute number of DOFs on an interface
    fn compute_interface_dofs(sub_a: &Subdomain, sub_b: &Subdomain, dim: usize) -> usize {
        let mut size = 1;
        for d in 0..3 {
            if d != dim {
                size *= sub_a.end[d].min(sub_b.end[d]) - sub_a.start[d].max(sub_b.start[d]);
            }
        }
        size
    }

    /// Set source field for the problem
    pub fn set_source(&mut self, global_source: &WaveArray<Complex64>) {
        for local in &mut self.local_domains {
            let subdomain = &local.subdomain;
            let local_shape = local.solution.lock().unwrap().shape_tuple();

            for i in 0..local_shape.0 {
                for j in 0..local_shape.1 {
                    for k in 0..local_shape.2 {
                        let gi = subdomain.start[0] + i;
                        let gj = subdomain.start[1] + j;
                        let gk = subdomain.start[2] + k;

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

        // Setup interface DOFs for each subdomain
        self.setup_interface_dofs();
    }

    /// Setup interface DOF mapping for each subdomain
    fn setup_interface_dofs(&mut self) {
        for interface in &self.interfaces {
            // Add DOFs to subdomain A (positive sign)
            self.add_interface_dofs_to_subdomain(
                interface.subdomain_a,
                interface,
                1.0,
            );

            // Add DOFs to subdomain B (negative sign for jump)
            self.add_interface_dofs_to_subdomain(
                interface.subdomain_b,
                interface,
                -1.0,
            );
        }
    }

    /// Add interface DOFs to a subdomain
    fn add_interface_dofs_to_subdomain(
        &mut self,
        subdomain_id: usize,
        interface: &Interface,
        sign: f64,
    ) {
        let local = &mut self.local_domains[subdomain_id];
        let subdomain = &local.subdomain;

        // Determine which face of the subdomain corresponds to this interface
        let is_start_face = interface.position == subdomain.start[interface.dimension];
        let is_end_face = interface.position == subdomain.end[interface.dimension];

        if !is_start_face && !is_end_face {
            return; // This subdomain doesn't touch this interface
        }

        // Generate DOFs on the interface face
        let mut dof_index = 0;
        match interface.dimension {
            0 => {
                // x-normal interface
                let i_local = if is_end_face {
                    subdomain.end[0] - subdomain.start[0] - 1
                } else {
                    0
                };
                for j in 0..(subdomain.end[1] - subdomain.start[1]) {
                    for k in 0..(subdomain.end[2] - subdomain.start[2]) {
                        local.interface_dofs.push(InterfaceDOF {
                            interface_id: interface.id,
                            local_dof: dof_index,
                            local_pos: [i_local, j, k],
                            sign,
                        });
                        dof_index += 1;
                    }
                }
            }
            1 => {
                // y-normal interface
                let j_local = if is_end_face {
                    subdomain.end[1] - subdomain.start[1] - 1
                } else {
                    0
                };
                for i in 0..(subdomain.end[0] - subdomain.start[0]) {
                    for k in 0..(subdomain.end[2] - subdomain.start[2]) {
                        local.interface_dofs.push(InterfaceDOF {
                            interface_id: interface.id,
                            local_dof: dof_index,
                            local_pos: [i, j_local, k],
                            sign,
                        });
                        dof_index += 1;
                    }
                }
            }
            2 => {
                // z-normal interface
                let k_local = if is_end_face {
                    subdomain.end[2] - subdomain.start[2] - 1
                } else {
                    0
                };
                for i in 0..(subdomain.end[0] - subdomain.start[0]) {
                    for j in 0..(subdomain.end[1] - subdomain.start[1]) {
                        local.interface_dofs.push(InterfaceDOF {
                            interface_id: interface.id,
                            local_dof: dof_index,
                            local_pos: [i, j, k_local],
                            sign,
                        });
                        dof_index += 1;
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    /// Solve using FETI method
    pub fn solve(&self, max_iterations: usize, tolerance: f64) -> WaveArray<Complex64> {
        println!("FETI: Solving dual interface problem...");

        // Step 1: Solve local problems without interface constraints (get particular solutions)
        self.solve_local_problems_unconstrained();

        // Step 2: Compute interface residual (jump)
        let interface_residual = self.compute_interface_residual();

        // Step 3: Solve dual interface problem for Lagrange multipliers
        // This is typically done with conjugate gradient or other iterative method
        self.solve_dual_problem(&interface_residual, max_iterations, tolerance);

        // Step 4: Correct local solutions using Lagrange multipliers
        self.apply_interface_corrections();

        // Step 5: Gather global solution
        self.gather_solution()
    }

    /// Solve local problems without interface constraints
    fn solve_local_problems_unconstrained(&self) {
        self.local_domains
            .par_iter()
            .for_each(|local| {
                let iter_config = IterationConfig {
                    max_iterations: 500,
                    threshold: 1e-6,
                    alpha: 0.75,
                    full_residuals: false,
                };

                let result = preconditioned_richardson(&local.solver, &local.source, iter_config);

                let mut solution = local.solution.lock().unwrap();
                *solution = result.field;
            });
    }

    /// Compute interface residual (jump in solution)
    fn compute_interface_residual(&self) -> Vec<Complex64> {
        let total_dofs: usize = self.interfaces.iter().map(|i| i.num_dofs).sum();
        let mut residual = vec![Complex::new(0.0, 0.0); total_dofs];

        let mut dof_offset = 0;
        for interface in &self.interfaces {
            // Get solutions from both subdomains
            let sol_a = self.local_domains[interface.subdomain_a]
                .solution
                .lock()
                .unwrap();
            let sol_b = self.local_domains[interface.subdomain_b]
                .solution
                .lock()
                .unwrap();

            // Compute jump [u] = u_a - u_b on interface
            for dof in &self.local_domains[interface.subdomain_a].interface_dofs {
                if dof.interface_id == interface.id {
                    let pos = dof.local_pos;
                    let val_a = sol_a.data[[pos[0], pos[1], pos[2]]];

                    // Find corresponding DOF in subdomain B
                    for dof_b in &self.local_domains[interface.subdomain_b].interface_dofs {
                        if dof_b.interface_id == interface.id && dof_b.local_dof == dof.local_dof {
                            let pos_b = dof_b.local_pos;
                            let val_b = sol_b.data[[pos_b[0], pos_b[1], pos_b[2]]];

                            residual[dof_offset + dof.local_dof] = val_a - val_b;
                            break;
                        }
                    }
                }
            }

            dof_offset += interface.num_dofs;
        }

        residual
    }

    /// Solve dual problem using simple iterative method (simplified FETI)
    fn solve_dual_problem(
        &self,
        _residual: &[Complex64],
        max_iterations: usize,
        tolerance: f64,
    ) {
        // Simplified dual solve: iteratively reduce interface jump
        // In full FETI, this would be a preconditioned conjugate gradient on the dual operator
        
        for iter in 0..max_iterations {
            let current_residual = self.compute_interface_residual();
            let residual_norm: f64 = current_residual
                .iter()
                .map(|r| r.norm_sqr())
                .sum::<f64>()
                .sqrt();

            if iter % 10 == 0 {
                println!("FETI dual iteration {}: residual = {:.2e}", iter, residual_norm);
            }

            if residual_norm < tolerance {
                println!(
                    "FETI converged after {} iterations (residual: {:.2e})",
                    iter + 1,
                    residual_norm
                );
                break;
            }

            // Update Lagrange multipliers to reduce jump
            // λ_new = λ_old + α * residual (simplified update)
            let alpha = 0.5 * self.dual_scaling;
            let mut lambda = self.lagrange_multipliers.lock().unwrap();
            for (l, r) in lambda.iter_mut().zip(current_residual.iter()) {
                *l += alpha * r;
            }
        }
    }

    /// Apply interface corrections using Lagrange multipliers
    fn apply_interface_corrections(&self) {
        let lambda = self.lagrange_multipliers.lock().unwrap();

        self.local_domains.par_iter().for_each(|local| {
            let mut solution = local.solution.lock().unwrap();

            // Apply correction from interface forces
            for dof in &local.interface_dofs {
                let interface = &self.interfaces[dof.interface_id];
                let dof_offset: usize = self.interfaces[..dof.interface_id]
                    .iter()
                    .map(|i| i.num_dofs)
                    .sum();

                let lambda_value = lambda[dof_offset + dof.local_dof];
                let pos = dof.local_pos;

                // Apply correction with proper sign
                let correction = dof.sign * lambda_value * 0.5;
                solution.data[[pos[0], pos[1], pos[2]]] += correction;
            }
        });
    }

    /// Gather global solution from local subdomains
    fn gather_solution(&self) -> WaveArray<Complex64> {
        let mut global = WaveArray::zeros(self.global_domain.shape);

        for local in &self.local_domains {
            let solution = local.solution.lock().unwrap();
            let subdomain = &local.subdomain;

            for i in 0..(subdomain.end[0] - subdomain.start[0]) {
                for j in 0..(subdomain.end[1] - subdomain.start[1]) {
                    for k in 0..(subdomain.end[2] - subdomain.start[2]) {
                        let gi = subdomain.start[0] + i;
                        let gj = subdomain.start[1] + j;
                        let gk = subdomain.start[2] + k;

                        global.data[[gi, gj, gk]] = solution.data[[i, j, k]];
                    }
                }
            }
        }

        global
    }
}

/// Solve Helmholtz equation using FETI domain decomposition
pub fn solve_helmholtz_feti(
    global_domain: HelmholtzDomain,
    source: WaveArray<Complex64>,
    num_subdomains: (usize, usize, usize),
    max_iterations: usize,
    tolerance: f64,
) -> WaveArray<Complex64> {
    println!(
        "Solving Helmholtz equation using FETI method with {} subdomains...",
        num_subdomains.0 * num_subdomains.1 * num_subdomains.2
    );

    let mut solver = FETIAcousticSolver::new(global_domain, num_subdomains);
    solver.set_source(&source);

    solver.solve(max_iterations, tolerance)
}

/// Analytical solution utilities for validation
pub mod analytical {
    use super::*;
    use special::Bessel;

    /// Compute analytical solution for circular domain using Bessel functions
    /// u(r,θ) = Σ Jₙ(kₙₘ r)[Aₙₘcos(nθ) + Bₙₘsin(nθ)]
    pub fn circular_cavity_2d(
        nx: usize,
        ny: usize,
        radius: f64,
        k: f64,
        n_modes: usize,
    ) -> WaveArray<Complex64> {
        let mut field = WaveArray::zeros((nx, ny, 1));
        let center_x = (nx as f64) / 2.0;
        let center_y = (ny as f64) / 2.0;
        let pixel_size = 2.0 * radius / (nx.min(ny) as f64);

        // Sum over Bessel function modes
        for n in 0..=n_modes {
            for m in 1..=3 {
                // Use first few roots of Bessel function
                let k_nm = bessel_zero(n, m) / radius;

                // Check if this mode is resonant
                if (k - k_nm).abs() < 0.2 * k {
                    let amplitude = 1.0 / ((n + 1) * m) as f64;

                    for i in 0..nx {
                        for j in 0..ny {
                            let x = (i as f64 - center_x) * pixel_size;
                            let y = (j as f64 - center_y) * pixel_size;
                            let r = (x * x + y * y).sqrt();
                            let theta = y.atan2(x);

                            if r <= radius {
                                let bessel_val = j_n(n, k_nm * r);
                                let angular = (n as f64 * theta).cos();
                                let value = amplitude * bessel_val * angular;
                                field.data[[i, j, 0]] += Complex::new(value, 0.0);
                            }
                        }
                    }
                }
            }
        }

        field
    }

    /// Compute m-th zero of n-th Bessel function (approximation)
    fn bessel_zero(n: usize, m: usize) -> f64 {
        // Approximate zeros using McMahon's asymptotic expansion
        let beta = (m as f64 + 0.5 * n as f64 - 0.25) * PI;
        let mu = 4.0 * (n * n) as f64;
        beta + (mu - 1.0) / (8.0 * beta)
            - 4.0 * (mu - 1.0) * (7.0 * mu - 31.0) / (3.0 * (8.0 * beta).powi(3))
    }

    /// Bessel function of the first kind J_n(x)
    fn j_n(n: usize, x: f64) -> f64 {
        if x.abs() < 1e-10 {
            if n == 0 {
                1.0
            } else {
                0.0
            }
        } else {
            x.bessel_j(n as f64)
        }
    }

    /// Compute spherical Bessel function j_n(x) (for 3D validation)
    pub fn spherical_bessel_j(n: usize, x: f64) -> f64 {
        if x.abs() < 1e-10 {
            if n == 0 {
                1.0
            } else {
                0.0
            }
        } else {
            // j_n(x) = sqrt(π/(2x)) * J_{n+1/2}(x)
            let jn_half = x.bessel_j(n as f64 + 0.5);
            (PI / (2.0 * x)).sqrt() * jn_half
        }
    }

    /// Compute spherical harmonic Y_n^m (simplified, real part only)
    pub fn spherical_harmonic(n: usize, m: i32, theta: f64, phi: f64) -> f64 {
        // Simplified implementation for low orders
        match (n, m) {
            (0, 0) => 0.5 / PI.sqrt(),
            (1, 0) => (3.0 / (4.0 * PI)).sqrt() * theta.cos(),
            (1, 1) => (3.0 / (8.0 * PI)).sqrt() * theta.sin() * phi.cos(),
            (1, -1) => (3.0 / (8.0 * PI)).sqrt() * theta.sin() * phi.sin(),
            _ => 0.0, // Would need full associated Legendre polynomials
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_feti_solver_creation() {
        let shape = (32, 32, 32);
        let permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

        let domain = HelmholtzDomain::new(
            permittivity,
            0.1,
            0.5,
            [false, false, false],
            [[2, 2], [2, 2], [2, 2]],
        );

        let solver = FETIAcousticSolver::new(domain, (2, 2, 2));

        assert_eq!(solver.local_domains.len(), 8);
        assert!(solver.interfaces.len() > 0);
    }

    #[test]
    fn test_interface_creation() {
        let shape = (32, 32, 32);
        let permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

        let domain = HelmholtzDomain::new(
            permittivity,
            0.1,
            0.5,
            [false, false, false],
            [[0, 0], [0, 0], [0, 0]],
        );

        let solver = FETIAcousticSolver::new(domain, (2, 2, 2));

        // With (2,2,2) decomposition, we should have 12 interfaces
        // 3 in x-direction, 3 in y-direction, 3 in z-direction per layer
        assert_eq!(solver.interfaces.len(), 12);
    }

    #[test]
    fn test_bessel_zero_approximation() {
        // Test first zero of J_0
        let z01 = analytical::bessel_zero(0, 1);
        // Known value is approximately 2.4048
        assert!((z01 - 2.4048).abs() < 0.1);

        // Test first zero of J_1
        let z11 = analytical::bessel_zero(1, 1);
        // Known value is approximately 3.8317
        assert!((z11 - 3.8317).abs() < 0.1);
    }

    #[test]
    fn test_spherical_bessel() {
        // j_0(x) = sin(x)/x
        let x = 1.0;
        let j0 = analytical::spherical_bessel_j(0, x);
        let expected = x.sin() / x;
        assert!((j0 - expected).abs() < 1e-6);
    }

    #[test]
    fn test_circular_analytical_solution() {
        let nx = 64;
        let ny = 64;
        let radius = 1.0;

        // Use first mode
        let k = analytical::bessel_zero(0, 1) / radius;

        let field = analytical::circular_cavity_2d(nx, ny, radius, k, 2);

        // Check field is not zero
        assert!(field.norm_squared() > 0.0);

        // Check that field is zero outside the circle
        let center = nx / 2;
        let pixel_size = 2.0 * radius / (nx as f64);
        let edge_r = (nx as f64) / 2.0 * pixel_size * 1.1;

        let edge_val = field.data[[0, center, 0]].norm();
        assert!(edge_val < 0.1); // Should be small outside circle
    }
}
