//! Sparse array implementation for source terms

use crate::engine::array::{Complex64, WaveArray};

/// Sparse array for representing source terms
#[derive(Debug, Clone)]
pub struct SparseArray {
    /// The data at specific positions
    pub data: Vec<WaveArray<Complex64>>,
    /// The positions where data is located
    pub positions: Vec<[usize; 3]>,
    /// The overall shape of the array
    pub shape: (usize, usize, usize),
}

impl SparseArray {
    /// Create a new sparse array
    pub fn new(
        data: Vec<WaveArray<Complex64>>,
        positions: Vec<[usize; 3]>,
        shape: (usize, usize, usize),
    ) -> Self {
        assert_eq!(
            data.len(),
            positions.len(),
            "Data and positions must have same length"
        );
        Self {
            data,
            positions,
            shape,
        }
    }

    /// Create an empty sparse array
    pub fn empty(shape: (usize, usize, usize)) -> Self {
        Self {
            data: Vec::new(),
            positions: Vec::new(),
            shape,
        }
    }

    /// Convert to a dense array
    pub fn to_dense(&self) -> WaveArray<Complex64> {
        let mut dense = WaveArray::zeros(self.shape);

        for (data, pos) in self.data.iter().zip(self.positions.iter()) {
            // Add the source data at the specified position
            let data_shape = data.shape();
            for i in 0..data_shape[0] {
                for j in 0..data_shape[1] {
                    for k in 0..data_shape[2] {
                        let global_i = pos[0] + i;
                        let global_j = pos[1] + j;
                        let global_k = pos[2] + k;

                        if global_i < self.shape.0
                            && global_j < self.shape.1
                            && global_k < self.shape.2
                        {
                            dense.data[[global_i, global_j, global_k]] += data.data[[i, j, k]];
                        }
                    }
                }
            }
        }

        dense
    }

    /// Get the number of non-zero blocks
    pub fn nnz(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_array_creation() {
        let data = vec![WaveArray::from_scalar((2, 2, 2), Complex64::new(1.0, 0.0))];
        let positions = vec![[5, 5, 5]];
        let sparse = SparseArray::new(data, positions, (10, 10, 10));

        assert_eq!(sparse.nnz(), 1);
        assert_eq!(sparse.shape, (10, 10, 10));
    }

    #[test]
    fn test_sparse_to_dense() {
        let data = vec![WaveArray::from_scalar((1, 1, 1), Complex64::new(5.0, 0.0))];
        let positions = vec![[2, 3, 4]];
        let sparse = SparseArray::new(data, positions, (5, 5, 5));

        let dense = sparse.to_dense();
        assert_eq!(dense.data[[2, 3, 4]], Complex64::new(5.0, 0.0));
        assert_eq!(dense.data[[0, 0, 0]], Complex64::new(0.0, 0.0));
    }
}
