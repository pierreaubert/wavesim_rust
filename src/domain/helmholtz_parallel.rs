//! Parallel Helmholtz equation solver using domain decomposition
//!
//! This module implements a parallel version of the Helmholtz solver
//! that uses domain decomposition with Rayon for shared-memory parallelization.

use crate::domain::domain_trait::Domain;
use crate::domain::helmholtz::HelmholtzDomain;
use crate::domain_decomposition::{DomainDecomposition, Subdomain};
use crate::engine::array::{Complex64, WaveArray};
use crate::engine::operations::multiply;
use num_traits::Zero;
use rayon::prelude::*;
use std::sync::Arc;

/// Parallel Helmholtz domain solver with domain decomposition
#[derive(Debug, Clone)]
pub struct ParallelHelmholtzDomain {
    /// Base Helmholtz domain (for shared parameters)
    base_domain: Arc<HelmholtzDomain>,
    /// Domain decomposition
    decomposition: Arc<DomainDecomposition>,
    /// Local permittivity arrays for each subdomain
    #[allow(dead_code)]
    local_permittivity: Vec<WaveArray<Complex64>>,
    /// Local B_scat arrays for each subdomain
    local_b_scat: Vec<WaveArray<Complex64>>,
    /// Overlap regions for boundary exchange
    overlap_size: usize,
}

impl ParallelHelmholtzDomain {
    /// Create a new parallel Helmholtz domain
    pub fn new(
        permittivity: WaveArray<Complex64>,
        pixel_size: f64,
        wavelength: f64,
        periodic: [bool; 3],
        boundary_widths: [[usize; 2]; 3],
        num_subdomains: (usize, usize, usize),
    ) -> Self {
        // Create base domain
        let base_domain = Arc::new(HelmholtzDomain::new(
            permittivity.clone(),
            pixel_size,
            wavelength,
            periodic,
            boundary_widths,
        ));

        // Create domain decomposition
        let shape = permittivity.shape_tuple();
        let overlap_size = 2; // Ghost cells for boundary exchange
        let decomposition = Arc::new(DomainDecomposition::new(
            shape,
            num_subdomains,
            overlap_size,
        ));

        // Create local arrays for each subdomain
        let (local_permittivity, local_b_scat) =
            Self::create_local_arrays(&permittivity, &base_domain, &decomposition, overlap_size);

        Self {
            base_domain,
            decomposition,
            local_permittivity,
            local_b_scat,
            overlap_size,
        }
    }

    /// Create local arrays for each subdomain
    fn create_local_arrays(
        permittivity: &WaveArray<Complex64>,
        base_domain: &HelmholtzDomain,
        decomposition: &DomainDecomposition,
        overlap_size: usize,
    ) -> (Vec<WaveArray<Complex64>>, Vec<WaveArray<Complex64>>) {
        let mut local_permittivity = Vec::new();
        let mut local_b_scat = Vec::new();

        for subdomain in &decomposition.subdomains {
            // Calculate local shape with overlap
            let local_shape = Self::get_local_shape_with_overlap(subdomain, overlap_size);

            // Extract local permittivity
            let mut local_perm = WaveArray::zeros(local_shape);
            Self::extract_local_data(permittivity, &mut local_perm, subdomain, overlap_size);

            // Create local B_scat
            let mut local_b = local_perm.clone();
            HelmholtzDomain::prepare_scattering_potential(
                &mut local_b,
                base_domain.k02,
                base_domain.shift,
                base_domain.scale_factor,
            );

            local_permittivity.push(local_perm);
            local_b_scat.push(local_b);
        }

        (local_permittivity, local_b_scat)
    }

    /// Get local shape including overlap regions
    fn get_local_shape_with_overlap(
        subdomain: &Subdomain,
        overlap_size: usize,
    ) -> (usize, usize, usize) {
        (
            subdomain.end[0] - subdomain.start[0] + 2 * overlap_size,
            subdomain.end[1] - subdomain.start[1] + 2 * overlap_size,
            subdomain.end[2] - subdomain.start[2] + 2 * overlap_size,
        )
    }

    /// Extract local data from global array
    fn extract_local_data(
        global: &WaveArray<Complex64>,
        local: &mut WaveArray<Complex64>,
        subdomain: &Subdomain,
        overlap_size: usize,
    ) {
        let global_shape = global.shape_tuple();

        for i in 0..local.shape()[0] {
            for j in 0..local.shape()[1] {
                for k in 0..local.shape()[2] {
                    // Map local indices to global indices
                    let gi = (subdomain.start[0] as isize + i as isize - overlap_size as isize)
                        .max(0)
                        .min(global_shape.0 as isize - 1) as usize;
                    let gj = (subdomain.start[1] as isize + j as isize - overlap_size as isize)
                        .max(0)
                        .min(global_shape.1 as isize - 1) as usize;
                    let gk = (subdomain.start[2] as isize + k as isize - overlap_size as isize)
                        .max(0)
                        .min(global_shape.2 as isize - 1) as usize;

                    local.data[[i, j, k]] = global.data[[gi, gj, gk]];
                }
            }
        }
    }

    /// Combine local data back to global array
    fn combine_local_data(
        local_arrays: &[WaveArray<Complex64>],
        global: &mut WaveArray<Complex64>,
        decomposition: &DomainDecomposition,
        overlap_size: usize,
    ) {
        // Zero out global array
        global.fill(Complex64::zero());

        for (idx, subdomain) in decomposition.subdomains.iter().enumerate() {
            let local = &local_arrays[idx];

            // Copy interior points (excluding overlap)
            for i in overlap_size..local.shape()[0] - overlap_size {
                for j in overlap_size..local.shape()[1] - overlap_size {
                    for k in overlap_size..local.shape()[2] - overlap_size {
                        let gi = subdomain.start[0] + i - overlap_size;
                        let gj = subdomain.start[1] + j - overlap_size;
                        let gk = subdomain.start[2] + k - overlap_size;

                        if gi < subdomain.end[0] && gj < subdomain.end[1] && gk < subdomain.end[2] {
                            global.data[[gi, gj, gk]] = local.data[[i, j, k]];
                        }
                    }
                }
            }
        }
    }

    /// Exchange boundary data between neighboring subdomains
    #[allow(dead_code)]
    fn exchange_boundaries(
        local_arrays: &mut [WaveArray<Complex64>],
        decomposition: &DomainDecomposition,
        overlap_size: usize,
    ) {
        // Create a copy for reading (to avoid race conditions)
        let arrays_copy: Vec<_> = local_arrays.iter().cloned().collect();

        // Process sequentially to avoid mutable borrow issues
        for (idx, subdomain) in decomposition.subdomains.iter().enumerate() {
            // Exchange with each neighbor
            if let Some(left_idx) = subdomain.neighbors.left {
                Self::copy_boundary_data(
                    &arrays_copy[left_idx],
                    &mut local_arrays[idx],
                    BoundaryDirection::FromLeft,
                    overlap_size,
                );
            }

            if let Some(right_idx) = subdomain.neighbors.right {
                Self::copy_boundary_data(
                    &arrays_copy[right_idx],
                    &mut local_arrays[idx],
                    BoundaryDirection::FromRight,
                    overlap_size,
                );
            }

            // Similar for other directions...
        }
    }

    /// Copy boundary data between subdomains
    #[allow(dead_code)]
    fn copy_boundary_data(
        source: &WaveArray<Complex64>,
        target: &mut WaveArray<Complex64>,
        direction: BoundaryDirection,
        overlap_size: usize,
    ) {
        let shape = target.shape_tuple();

        match direction {
            BoundaryDirection::FromLeft => {
                // Copy from right boundary of source to left ghost cells of target
                for j in 0..shape.1 {
                    for k in 0..shape.2 {
                        for i in 0..overlap_size {
                            target.data[[i, j, k]] =
                                source.data[[source.shape()[0] - 2 * overlap_size + i, j, k]];
                        }
                    }
                }
            }
            BoundaryDirection::FromRight => {
                // Copy from left boundary of source to right ghost cells of target
                for j in 0..shape.1 {
                    for k in 0..shape.2 {
                        for i in 0..overlap_size {
                            target.data[[shape.0 - overlap_size + i, j, k]] =
                                source.data[[overlap_size + i, j, k]];
                        }
                    }
                }
            } // Add other directions as needed
        }
    }

    /// Apply medium operator in parallel across subdomains
    pub fn medium_parallel(&self, x: &WaveArray<Complex64>, out: &mut WaveArray<Complex64>) {
        // Split input into local arrays
        let mut local_x = Vec::new();
        for subdomain in &self.decomposition.subdomains {
            let mut local = WaveArray::zeros(Self::get_local_shape_with_overlap(
                subdomain,
                self.overlap_size,
            ));
            Self::extract_local_data(x, &mut local, subdomain, self.overlap_size);
            local_x.push(local);
        }

        // Process each subdomain in parallel
        let local_results: Vec<_> = local_x
            .par_iter()
            .zip(self.local_b_scat.par_iter())
            .map(|(local_input, local_b)| multiply(local_input, local_b))
            .collect();

        // Combine results
        Self::combine_local_data(&local_results, out, &self.decomposition, self.overlap_size);
    }

    /// Apply propagator in parallel
    pub fn propagator_parallel(&self, x: &WaveArray<Complex64>, out: &mut WaveArray<Complex64>) {
        // For the propagator, we need global FFT operations
        // So we use the base domain's method but can parallelize the FFT itself
        self.base_domain.propagator(x, out);
    }
}

/// Direction for boundary data exchange
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
enum BoundaryDirection {
    FromLeft,
    FromRight,
    // FromFront,
    // FromBack,
    // FromBottom,
    // FromTop,
}

/// Solve Helmholtz equation using parallel domain decomposition
pub fn solve_helmholtz_parallel(
    domain: &ParallelHelmholtzDomain,
    source: WaveArray<Complex64>,
    max_iterations: usize,
    tolerance: f64,
) -> WaveArray<Complex64> {
    let mut x = source.clone();
    let mut residual = WaveArray::zeros(x.shape_tuple());

    println!(
        "Solving Helmholtz equation with {} subdomains...",
        domain.decomposition.num_subdomains()
    );

    for iter in 0..max_iterations {
        // Apply medium operator in parallel
        domain.medium_parallel(&x, &mut residual);

        // Apply propagator (global operation)
        domain.propagator_parallel(&residual, &mut x);

        // Check convergence
        let error = residual.norm_squared().sqrt();
        if error < tolerance {
            println!(
                "Converged after {} iterations (error: {:.2e})",
                iter + 1,
                error
            );
            break;
        }

        if iter % 10 == 0 {
            println!("Iteration {}: error = {:.2e}", iter, error);
        }
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_parallel_helmholtz_creation() {
        let shape = (50, 50, 50);
        let permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

        let domain = ParallelHelmholtzDomain::new(
            permittivity,
            0.1, // pixel_size
            0.5, // wavelength
            [false, false, false],
            [[1, 1], [1, 1], [1, 1]],
            (2, 2, 2), // 8 subdomains
        );

        assert_eq!(domain.decomposition.num_subdomains(), 8);
        assert_eq!(domain.local_permittivity.len(), 8);
        assert_eq!(domain.local_b_scat.len(), 8);
    }

    #[test]
    fn test_local_shape_calculation() {
        let subdomain = Subdomain {
            id: 0,
            start: [0, 0, 0],
            end: [25, 25, 25],
            neighbors: Default::default(),
        };

        let shape = ParallelHelmholtzDomain::get_local_shape_with_overlap(&subdomain, 2);
        assert_eq!(shape, (29, 29, 29)); // 25 + 2*2 = 29
    }
}
