//! Domain decomposition for parallel FDTD simulations
//!
//! This module provides functionality for splitting the computational domain
//! into subdomains for parallel processing using Rayon.

use crate::engine::array::{Complex64, WaveArray};
use rayon::prelude::*;

/// Information about a single subdomain
#[derive(Debug, Clone)]
pub struct Subdomain {
    /// ID of this subdomain
    pub id: usize,
    /// Starting indices in the global domain
    pub start: [usize; 3],
    /// Ending indices in the global domain (exclusive)
    pub end: [usize; 3],
    /// Indices of neighboring subdomains
    pub neighbors: Neighbors,
}

/// Neighbor information for a subdomain
#[derive(Debug, Clone, Default)]
pub struct Neighbors {
    pub left: Option<usize>,   // -x direction
    pub right: Option<usize>,  // +x direction
    pub front: Option<usize>,  // -y direction
    pub back: Option<usize>,   // +y direction
    pub bottom: Option<usize>, // -z direction
    pub top: Option<usize>,    // +z direction
}

/// Domain decomposition manager
#[derive(Debug)]
pub struct DomainDecomposition {
    /// Global domain shape
    pub global_shape: (usize, usize, usize),
    /// Number of subdomains in each direction
    pub decomposition: (usize, usize, usize),
    /// List of all subdomains
    pub subdomains: Vec<Subdomain>,
    /// Ghost cell thickness (for boundary exchange)
    pub ghost_cells: usize,
}

impl DomainDecomposition {
    /// Create a new domain decomposition
    pub fn new(
        global_shape: (usize, usize, usize),
        decomposition: (usize, usize, usize),
        ghost_cells: usize,
    ) -> Self {
        let subdomains = Self::create_subdomains(global_shape, decomposition);

        Self {
            global_shape,
            decomposition,
            subdomains,
            ghost_cells,
        }
    }

    /// Create subdomains with proper neighbor information
    fn create_subdomains(
        global_shape: (usize, usize, usize),
        decomposition: (usize, usize, usize),
    ) -> Vec<Subdomain> {
        let mut subdomains = Vec::new();

        // Calculate subdomain sizes
        let subdomain_size = [
            global_shape.0 / decomposition.0,
            global_shape.1 / decomposition.1,
            global_shape.2 / decomposition.2,
        ];

        // Create subdomains
        for i in 0..decomposition.0 {
            for j in 0..decomposition.1 {
                for k in 0..decomposition.2 {
                    let id = k + j * decomposition.2 + i * decomposition.1 * decomposition.2;

                    let start = [
                        i * subdomain_size[0],
                        j * subdomain_size[1],
                        k * subdomain_size[2],
                    ];

                    let end = [
                        if i == decomposition.0 - 1 {
                            global_shape.0
                        } else {
                            (i + 1) * subdomain_size[0]
                        },
                        if j == decomposition.1 - 1 {
                            global_shape.1
                        } else {
                            (j + 1) * subdomain_size[1]
                        },
                        if k == decomposition.2 - 1 {
                            global_shape.2
                        } else {
                            (k + 1) * subdomain_size[2]
                        },
                    ];

                    // Determine neighbors
                    let neighbors = Neighbors {
                        left: if i > 0 {
                            Some(
                                k + j * decomposition.2
                                    + (i - 1) * decomposition.1 * decomposition.2,
                            )
                        } else {
                            None
                        },
                        right: if i < decomposition.0 - 1 {
                            Some(
                                k + j * decomposition.2
                                    + (i + 1) * decomposition.1 * decomposition.2,
                            )
                        } else {
                            None
                        },
                        front: if j > 0 {
                            Some(
                                k + (j - 1) * decomposition.2
                                    + i * decomposition.1 * decomposition.2,
                            )
                        } else {
                            None
                        },
                        back: if j < decomposition.1 - 1 {
                            Some(
                                k + (j + 1) * decomposition.2
                                    + i * decomposition.1 * decomposition.2,
                            )
                        } else {
                            None
                        },
                        bottom: if k > 0 {
                            Some(
                                (k - 1)
                                    + j * decomposition.2
                                    + i * decomposition.1 * decomposition.2,
                            )
                        } else {
                            None
                        },
                        top: if k < decomposition.2 - 1 {
                            Some(
                                (k + 1)
                                    + j * decomposition.2
                                    + i * decomposition.1 * decomposition.2,
                            )
                        } else {
                            None
                        },
                    };

                    subdomains.push(Subdomain {
                        id,
                        start,
                        end,
                        neighbors,
                    });
                }
            }
        }

        subdomains
    }

    /// Get the subdomain for a given index
    pub fn get_subdomain(&self, index: usize) -> &Subdomain {
        &self.subdomains[index]
    }

    /// Get the number of subdomains
    pub fn num_subdomains(&self) -> usize {
        self.subdomains.len()
    }

    /// Check if indices are on a subdomain boundary
    pub fn is_boundary(&self, subdomain: &Subdomain, i: usize, j: usize, k: usize) -> bool {
        i == subdomain.start[0]
            || i == subdomain.end[0] - 1
            || j == subdomain.start[1]
            || j == subdomain.end[1] - 1
            || k == subdomain.start[2]
            || k == subdomain.end[2] - 1
    }
}

/// Boundary data for exchange between subdomains
#[derive(Debug, Clone)]
pub struct BoundaryData {
    pub subdomain_id: usize,
    pub face: Face,
    pub data: Vec<Complex64>,
}

/// Face identifier for boundary exchange
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Face {
    Left,   // -x
    Right,  // +x
    Front,  // -y
    Back,   // +y
    Bottom, // -z
    Top,    // +z
}

/// Exchange boundary data between subdomains
pub fn exchange_boundaries(
    _fields: &mut [WaveArray<Complex64>],
    decomposition: &DomainDecomposition,
) {
    // For shared memory, we can directly copy boundary data
    // In a distributed setting, this would involve MPI communication

    // This is a simplified version - in practice, we'd need to handle
    // each field component (Ex, Ey, Ez, Hx, Hy, Hz) separately

    // Using Rayon to parallelize boundary exchange
    rayon::scope(|s| {
        for _subdomain in &decomposition.subdomains {
            s.spawn(move |_| {
                // Extract and exchange boundary data
                // This would be implemented based on the specific FDTD stencil
            });
        }
    });
}

/// Parallel FDTD update using domain decomposition
pub fn parallel_fdtd_update<F>(decomposition: &DomainDecomposition, update_fn: F)
where
    F: Fn(usize, &Subdomain) + Send + Sync,
{
    // Use Rayon to parallelize across subdomains
    decomposition
        .subdomains
        .par_iter()
        .enumerate()
        .for_each(|(idx, subdomain)| {
            update_fn(idx, subdomain);
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_decomposition_creation() {
        let global_shape = (100, 100, 100);
        let decomposition = (2, 2, 2);
        let dd = DomainDecomposition::new(global_shape, decomposition, 1);

        assert_eq!(dd.num_subdomains(), 8);

        // Check first subdomain
        let first = dd.get_subdomain(0);
        assert_eq!(first.start, [0, 0, 0]);
        assert_eq!(first.end, [50, 50, 50]);

        // Check neighbors
        assert!(first.neighbors.left.is_none());
        assert!(first.neighbors.right.is_some());
        assert!(first.neighbors.front.is_none());
        assert!(first.neighbors.back.is_some());
        assert!(first.neighbors.bottom.is_none());
        assert!(first.neighbors.top.is_some());
    }

    #[test]
    fn test_boundary_detection() {
        let global_shape = (10, 10, 10);
        let decomposition = (2, 1, 1);
        let dd = DomainDecomposition::new(global_shape, decomposition, 1);

        let subdomain = dd.get_subdomain(0);

        // Test boundary detection
        assert!(dd.is_boundary(subdomain, 0, 5, 5)); // Left boundary
        assert!(dd.is_boundary(subdomain, 4, 5, 5)); // Right boundary
        assert!(!dd.is_boundary(subdomain, 2, 5, 5)); // Interior
    }
}
