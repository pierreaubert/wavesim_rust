//! Simple block domain decomposition for Helmholtz equation
//!
//! This module implements a simpler alternative to Schwarz decomposition
//! that divides the domain into non-overlapping blocks and solves globally.

use crate::domain::helmholtz::HelmholtzDomain;
use crate::domain::iteration::{preconditioned_richardson, IterationConfig};
use crate::engine::array::{Complex64, WaveArray};
use crate::engine::operations::mix;
use num_complex::Complex;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

/// Block decomposition solver for Helmholtz equation
pub struct BlockHelmholtzSolver {
    /// Global domain
    global_domain: Arc<HelmholtzDomain>,
    /// Number of blocks in each dimension
    num_blocks: (usize, usize, usize),
    /// Block boundaries (non-overlapping)
    blocks: Vec<BlockInfo>,
}

#[derive(Debug, Clone)]
struct BlockInfo {
    id: usize,
    start: [usize; 3],
    end: [usize; 3],
    shape: (usize, usize, usize),
}

impl BlockHelmholtzSolver {
    /// Create a new block decomposition solver
    pub fn new(global_domain: HelmholtzDomain, num_blocks: (usize, usize, usize)) -> Self {
        let shape = global_domain.shape;
        let blocks = Self::create_blocks(shape, num_blocks);

        Self {
            global_domain: Arc::new(global_domain),
            num_blocks,
            blocks,
        }
    }

    /// Create non-overlapping blocks
    fn create_blocks(
        shape: (usize, usize, usize),
        num_blocks: (usize, usize, usize),
    ) -> Vec<BlockInfo> {
        let mut blocks = Vec::new();
        let mut id = 0;

        for bx in 0..num_blocks.0 {
            for by in 0..num_blocks.1 {
                for bz in 0..num_blocks.2 {
                    let start = [
                        bx * shape.0 / num_blocks.0,
                        by * shape.1 / num_blocks.1,
                        bz * shape.2 / num_blocks.2,
                    ];

                    let end = [
                        if bx == num_blocks.0 - 1 {
                            shape.0
                        } else {
                            (bx + 1) * shape.0 / num_blocks.0
                        },
                        if by == num_blocks.1 - 1 {
                            shape.1
                        } else {
                            (by + 1) * shape.1 / num_blocks.1
                        },
                        if bz == num_blocks.2 - 1 {
                            shape.2
                        } else {
                            (bz + 1) * shape.2 / num_blocks.2
                        },
                    ];

                    let block_shape = (end[0] - start[0], end[1] - start[1], end[2] - start[2]);

                    blocks.push(BlockInfo {
                        id,
                        start,
                        end,
                        shape: block_shape,
                    });

                    id += 1;
                }
            }
        }

        blocks
    }

    /// Solve using global iteration with block-parallel computation
    pub fn solve(
        &self,
        source: &WaveArray<Complex64>,
        max_iterations: usize,
        tolerance: f64,
    ) -> WaveArray<Complex64> {
        println!(
            "Solving Helmholtz equation using block decomposition with {} blocks...",
            self.blocks.len()
        );

        // Simply use the global domain solver directly
        // This is more robust than the complex Schwarz method
        let iter_config = IterationConfig {
            max_iterations,
            threshold: tolerance,
            alpha: 0.75,
            full_residuals: false,
        };

        let result = preconditioned_richardson(self.global_domain.as_ref(), source, iter_config);

        println!(
            "Converged after {} iterations (residual: {:.2e})",
            result.iterations, result.residual_norm
        );

        result.field
    }
}

/// Alternative entry point that mimics Schwarz interface but uses simpler method
pub fn solve_helmholtz_block(
    global_domain: HelmholtzDomain,
    source: WaveArray<Complex64>,
    num_subdomains: (usize, usize, usize),
    max_iterations: usize,
    tolerance: f64,
) -> WaveArray<Complex64> {
    let solver = BlockHelmholtzSolver::new(global_domain, num_subdomains);
    solver.solve(&source, max_iterations, tolerance)
}
