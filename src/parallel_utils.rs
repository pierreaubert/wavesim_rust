//! Parallel utilities for optimized domain decomposition
//!
//! This module provides optimized utilities for boundary exchange
//! and parallel operations in domain decomposition methods.

use crate::engine::array::{Complex64, WaveArray};
use rayon::prelude::*;
use std::cell::UnsafeCell;
use std::sync::Arc;

/// Thread-safe buffer pool for reducing allocations
pub struct BufferPool {
    buffers: Vec<UnsafeCell<WaveArray<Complex64>>>,
}

unsafe impl Sync for BufferPool {}

impl BufferPool {
    /// Create a new buffer pool
    pub fn new(num_buffers: usize, shape: (usize, usize, usize)) -> Arc<Self> {
        let buffers = (0..num_buffers)
            .map(|_| UnsafeCell::new(WaveArray::zeros(shape)))
            .collect();

        Arc::new(Self { buffers })
    }

    /// Get a buffer from the pool
    pub fn get(&self, index: usize) -> &mut WaveArray<Complex64> {
        unsafe { &mut *self.buffers[index].get() }
    }
}

/// Optimized boundary exchange using double buffering
pub struct BoundaryExchange {
    /// Number of subdomains
    _num_subdomains: usize,
    /// Send buffers for each subdomain
    send_buffers: Vec<Vec<Vec<Complex64>>>,
    /// Receive buffers for each subdomain
    recv_buffers: Vec<Vec<Vec<Complex64>>>,
}

impl BoundaryExchange {
    /// Create a new boundary exchange manager
    pub fn new(num_subdomains: usize, max_boundary_size: usize) -> Self {
        let send_buffers = (0..num_subdomains)
            .map(|_| {
                (0..6)
                    .map(|_| Vec::with_capacity(max_boundary_size))
                    .collect()
            })
            .collect();

        let recv_buffers = (0..num_subdomains)
            .map(|_| {
                (0..6)
                    .map(|_| Vec::with_capacity(max_boundary_size))
                    .collect()
            })
            .collect();

        Self {
            _num_subdomains: num_subdomains,
            send_buffers,
            recv_buffers,
        }
    }

    /// Pack boundary data for sending
    pub fn pack_boundaries(
        &mut self,
        subdomain_id: usize,
        field: &WaveArray<Complex64>,
        shape: (usize, usize, usize),
    ) {
        // Clear previous data
        for buffer in &mut self.send_buffers[subdomain_id] {
            buffer.clear();
        }

        // Pack left boundary (face 0)
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                self.send_buffers[subdomain_id][0].push(field.data[[0, j, k]]);
            }
        }

        // Pack right boundary (face 1)
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                self.send_buffers[subdomain_id][1].push(field.data[[shape.0 - 1, j, k]]);
            }
        }

        // Pack front boundary (face 2)
        for i in 0..shape.0 {
            for k in 0..shape.2 {
                self.send_buffers[subdomain_id][2].push(field.data[[i, 0, k]]);
            }
        }

        // Pack back boundary (face 3)
        for i in 0..shape.0 {
            for k in 0..shape.2 {
                self.send_buffers[subdomain_id][3].push(field.data[[i, shape.1 - 1, k]]);
            }
        }

        // Pack bottom boundary (face 4)
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                self.send_buffers[subdomain_id][4].push(field.data[[i, j, 0]]);
            }
        }

        // Pack top boundary (face 5)
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                self.send_buffers[subdomain_id][5].push(field.data[[i, j, shape.2 - 1]]);
            }
        }
    }

    /// Exchange boundaries between neighbors
    pub fn exchange(&mut self, neighbor_map: &[(usize, usize, usize)]) {
        // Copy send buffers to receive buffers based on neighbor map
        // neighbor_map contains (subdomain_id, face_id, neighbor_subdomain_id)

        for &(subdomain, face, neighbor) in neighbor_map {
            let opposite_face = Self::opposite_face(face);

            // Copy data from neighbor's send buffer to our receive buffer
            self.recv_buffers[subdomain][face].clear();
            self.recv_buffers[subdomain][face]
                .extend_from_slice(&self.send_buffers[neighbor][opposite_face]);
        }
    }

    /// Unpack received boundary data
    pub fn unpack_boundaries(
        &self,
        subdomain_id: usize,
        field: &mut WaveArray<Complex64>,
        shape: (usize, usize, usize),
        ghost_width: usize,
    ) {
        let mut idx;

        // Unpack left ghost cells (from face 0)
        if !self.recv_buffers[subdomain_id][0].is_empty() {
            idx = 0;
            for j in 0..shape.1 {
                for k in 0..shape.2 {
                    for g in 0..ghost_width {
                        if g < shape.0 {
                            field.data[[g, j, k]] = self.recv_buffers[subdomain_id][0][idx];
                        }
                    }
                    idx += 1;
                }
            }
        }

        // Similarly for other faces...
    }

    /// Get opposite face index
    fn opposite_face(face: usize) -> usize {
        match face {
            0 => 1, // left -> right
            1 => 0, // right -> left
            2 => 3, // front -> back
            3 => 2, // back -> front
            4 => 5, // bottom -> top
            5 => 4, // top -> bottom
            _ => panic!("Invalid face index"),
        }
    }
}

/// Parallel reduction for computing norms and inner products
pub fn parallel_norm_squared(field: &WaveArray<Complex64>) -> f64 {
    let shape = field.shape_tuple();
    let chunk_size = shape.0 / rayon::current_num_threads().max(1);

    field
        .data
        .as_slice()
        .unwrap()
        .par_chunks(chunk_size * shape.1 * shape.2)
        .map(|chunk| chunk.iter().map(|&c| (c.conj() * c).re).sum::<f64>())
        .sum()
}

/// Parallel field update with optimal cache usage
pub fn parallel_field_update<F>(
    field: &mut WaveArray<Complex64>,
    _shape: (usize, usize, usize),
    update_fn: F,
) where
    F: Fn(usize, usize, usize, Complex64) -> Complex64 + Sync + Send,
{
    use ndarray::parallel::prelude::*;

    // Use ndarray's parallel iterator which handles the synchronization
    field
        .data
        .indexed_iter_mut()
        .par_bridge()
        .for_each(|((i, j, k), val)| {
            *val = update_fn(i, j, k, *val);
        });
}

/// Prefetch data for better cache performance
#[inline(always)]
pub fn prefetch_read<T>(_ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::_mm_prefetch;
        _mm_prefetch(_ptr as *const i8, 0);
    }
}

/// SIMD-optimized complex multiplication (when available)
#[cfg(target_arch = "x86_64")]
pub mod simd {
    use std::arch::x86_64::*;

    /// Multiply two complex arrays using AVX2 if available
    pub unsafe fn complex_multiply_avx2(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
        // Assumes complex numbers are stored as [real, imag, real, imag, ...]
        let chunks = len / 4; // Process 2 complex numbers at once

        for i in 0..chunks {
            let offset = i * 4;

            // Load 2 complex numbers from each array
            let a_vec = _mm256_loadu_pd(&a[offset]);
            let b_vec = _mm256_loadu_pd(&b[offset]);

            // Perform complex multiplication
            // (a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)

            // Shuffle to get [a.re, a.re, a.im, a.im]
            let a_re = _mm256_unpacklo_pd(a_vec, a_vec);
            let a_im = _mm256_unpackhi_pd(a_vec, a_vec);

            // Multiply
            let re_part = _mm256_mul_pd(a_re, b_vec);

            // Shuffle b to get [b.im, b.re, b.im, b.re]
            let b_swapped = _mm256_shuffle_pd(b_vec, b_vec, 0b0101);
            let im_part = _mm256_mul_pd(a_im, b_swapped);

            // Combine with appropriate signs
            let signs = _mm256_set_pd(1.0, -1.0, 1.0, -1.0);
            let im_signed = _mm256_mul_pd(im_part, signs);

            let result_vec = _mm256_add_pd(re_part, im_signed);

            // Store result
            _mm256_storeu_pd(&mut result[offset], result_vec);
        }

        // Handle remaining elements
        for i in (chunks * 4)..len {
            result[i] = a[i] * b[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_buffer_pool() {
        let pool = BufferPool::new(4, (10, 10, 10));

        let buffer1 = pool.get(0);
        buffer1.data[[0, 0, 0]] = Complex::new(1.0, 0.0);

        let buffer2 = pool.get(1);
        buffer2.data[[0, 0, 0]] = Complex::new(2.0, 0.0);

        assert_eq!(pool.get(0).data[[0, 0, 0]], Complex::new(1.0, 0.0));
        assert_eq!(pool.get(1).data[[0, 0, 0]], Complex::new(2.0, 0.0));
    }

    #[test]
    fn test_boundary_exchange() {
        let mut exchange = BoundaryExchange::new(2, 100);

        let field = WaveArray::from_scalar((10, 10, 10), Complex::new(1.0, 0.0));
        exchange.pack_boundaries(0, &field, (10, 10, 10));

        assert_eq!(exchange.send_buffers[0][0].len(), 100); // left face
        assert_eq!(exchange.send_buffers[0][1].len(), 100); // right face
    }
}
