//! Block array implementation for domain decomposition

use crate::engine::array::{ArraySlice, Complex64, WaveArray};
use rayon::prelude::*;

/// Block array for domain decomposition
#[derive(Debug, Clone)]
pub struct BlockArray {
    /// The blocks of the array
    pub blocks: Vec<Vec<Vec<WaveArray<Complex64>>>>,
    /// Number of blocks in each dimension
    pub n_blocks: [usize; 3],
    /// The overall shape of the array
    pub shape: (usize, usize, usize),
    /// Block boundaries for each dimension
    pub boundaries: Vec<Vec<usize>>,
}

impl BlockArray {
    /// Create a new block array from a single array
    pub fn from_array(array: WaveArray<Complex64>, n_blocks: [usize; 3]) -> Self {
        let shape = array.shape_tuple();
        let boundaries = Self::compute_boundaries(shape, n_blocks);

        // Split the array into blocks
        let mut blocks = vec![vec![vec![]; n_blocks[2]]; n_blocks[1]];
        for i in 0..n_blocks[0] {
            blocks.push(vec![vec![]; n_blocks[2]]);
            for j in 0..n_blocks[1] {
                for k in 0..n_blocks[2] {
                    let start = [boundaries[0][i], boundaries[1][j], boundaries[2][k]];
                    let stop = [
                        boundaries[0][i + 1],
                        boundaries[1][j + 1],
                        boundaries[2][k + 1],
                    ];

                    // Use the slice method from the trait
                    let block = array.slice(start, stop);
                    blocks[i][j].push(block);
                }
            }
        }

        Self {
            blocks,
            n_blocks,
            shape,
            boundaries,
        }
    }

    /// Compute block boundaries
    fn compute_boundaries(shape: (usize, usize, usize), n_blocks: [usize; 3]) -> Vec<Vec<usize>> {
        let mut boundaries = Vec::new();

        for dim in 0..3 {
            let size = match dim {
                0 => shape.0,
                1 => shape.1,
                2 => shape.2,
                _ => unreachable!(),
            };

            let n = n_blocks[dim];
            let block_size = size / n;
            let remainder = size % n;

            let mut bounds = vec![0];
            let mut pos = 0;
            for i in 0..n {
                pos += block_size;
                if i < remainder {
                    pos += 1;
                }
                bounds.push(pos);
            }

            boundaries.push(bounds);
        }

        boundaries
    }

    /// Get a specific block
    pub fn get_block(&self, i: usize, j: usize, k: usize) -> &WaveArray<Complex64> {
        &self.blocks[i][j][k]
    }

    /// Get a mutable reference to a specific block
    pub fn get_block_mut(&mut self, i: usize, j: usize, k: usize) -> &mut WaveArray<Complex64> {
        &mut self.blocks[i][j][k]
    }

    /// Gather all blocks into a single array
    pub fn gather(&self) -> WaveArray<Complex64> {
        let mut result = WaveArray::zeros(self.shape);

        for i in 0..self.n_blocks[0] {
            for j in 0..self.n_blocks[1] {
                for k in 0..self.n_blocks[2] {
                    let block = &self.blocks[i][j][k];
                    let start = [
                        self.boundaries[0][i],
                        self.boundaries[1][j],
                        self.boundaries[2][k],
                    ];

                    // Copy block data into result
                    let block_shape = block.shape();
                    for bi in 0..block_shape[0] {
                        for bj in 0..block_shape[1] {
                            for bk in 0..block_shape[2] {
                                result.data[[start[0] + bi, start[1] + bj, start[2] + bk]] =
                                    block.data[[bi, bj, bk]];
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Apply a function to each block in parallel
    pub fn par_map_inplace<F>(&mut self, f: F)
    where
        F: Fn(&mut WaveArray<Complex64>) + Sync + Send,
    {
        self.blocks.par_iter_mut().for_each(|row| {
            row.par_iter_mut().for_each(|col| {
                col.par_iter_mut().for_each(|block| {
                    f(block);
                });
            });
        });
    }

    /// Get block shape for a specific block
    pub fn block_shape(&self, i: usize, j: usize, k: usize) -> (usize, usize, usize) {
        self.blocks[i][j][k].shape_tuple()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_array_creation() {
        let array = WaveArray::from_scalar((10, 10, 10), Complex64::new(1.0, 0.0));
        let block_array = BlockArray::from_array(array, [2, 2, 2]);

        assert_eq!(block_array.n_blocks, [2, 2, 2]);
        assert_eq!(block_array.shape, (10, 10, 10));
    }

    #[test]
    fn test_block_boundaries() {
        let boundaries = BlockArray::compute_boundaries((10, 10, 10), [2, 2, 2]);

        assert_eq!(boundaries[0], vec![0, 5, 10]);
        assert_eq!(boundaries[1], vec![0, 5, 10]);
        assert_eq!(boundaries[2], vec![0, 5, 10]);
    }

    #[test]
    fn test_block_array_gather() {
        let array = WaveArray::from_scalar((10, 10, 10), Complex64::new(1.0, 0.0));
        let block_array = BlockArray::from_array(array.clone(), [2, 2, 2]);
        let gathered = block_array.gather();

        assert_eq!(gathered.shape(), array.shape());
        assert_eq!(gathered.data[[0, 0, 0]], array.data[[0, 0, 0]]);
    }
}
