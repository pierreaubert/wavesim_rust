//! Tests for basic engine operations, ported from wavesim_py/tests/test_engine.py

use approx::assert_relative_eq;
use num_complex::Complex;
use wavesim::engine::array::{Complex64, WaveArray};
use wavesim::engine::block::BlockArray;
use wavesim::engine::operations::*;
use wavesim::engine::sparse::SparseArray;

/// Helper function to generate test arrays with random data
fn random_array(shape: (usize, usize, usize)) -> WaveArray<Complex64> {
    let mut array = WaveArray::zeros(shape);
    let size = shape.0 * shape.1 * shape.2;

    for i in 0..size {
        let idx = (
            i / (shape.1 * shape.2),
            (i / shape.2) % shape.1,
            i % shape.2,
        );
        let real = ((i as f64) * 0.123).sin();
        let imag = ((i as f64) * 0.456).cos();
        array.data[[idx.0, idx.1, idx.2]] = Complex::new(real, imag);
    }

    array
}

/// Test basic array creation and operations
#[test]
fn test_array_creation() {
    let shape = (10, 10, 10);

    // Test zeros
    let zeros = WaveArray::<Complex64>::zeros(shape);
    assert_eq!(zeros.shape_tuple(), shape);
    assert_eq!(zeros.len(), 1000);
    for val in zeros.data.iter() {
        assert_eq!(*val, Complex::new(0.0, 0.0));
    }

    // Test from_scalar
    let scalar_val = Complex::new(2.5, -1.5);
    let scalar_array = WaveArray::from_scalar(shape, scalar_val);
    for val in scalar_array.data.iter() {
        assert_eq!(*val, scalar_val);
    }

    // Test clone
    let original = random_array(shape);
    let cloned = original.clone();
    assert_eq!(original.shape_tuple(), cloned.shape_tuple());
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                assert_eq!(original.data[[i, j, k]], cloned.data[[i, j, k]]);
            }
        }
    }
}

/// Test scale operation
#[test]
fn test_scale() {
    let shape = (5, 5, 5);
    let input = WaveArray::from_scalar(shape, Complex::new(2.0, 1.0));
    let mut output = WaveArray::zeros(shape);

    // Test scale without offset
    scale(Complex::new(3.0, 0.0), &input, None, &mut output);

    for val in output.data.iter() {
        assert_eq!(*val, Complex::new(6.0, 3.0));
    }

    // Test scale with offset
    scale(
        Complex::new(2.0, 0.0),
        &input,
        Some(Complex::new(1.0, 0.0)),
        &mut output,
    );

    for val in output.data.iter() {
        assert_eq!(*val, Complex::new(5.0, 2.0));
    }
}

/// Test mix operation
#[test]
fn test_mix() {
    let shape = (5, 5, 5);
    let a = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
    let b = WaveArray::from_scalar(shape, Complex::new(2.0, 0.0));
    let mut output = WaveArray::zeros(shape);

    mix(
        Complex::new(2.0, 0.0),
        &a,
        Complex::new(3.0, 0.0),
        &b,
        &mut output,
    );

    for val in output.data.iter() {
        assert_eq!(*val, Complex::new(8.0, 0.0));
    }
}

/// Test multiply operation
#[test]
fn test_multiply() {
    let shape = (4, 4, 4);
    let a = WaveArray::from_scalar(shape, Complex::new(2.0, 1.0));
    let b = WaveArray::from_scalar(shape, Complex::new(3.0, -1.0));

    let result = multiply(&a, &b);

    // (2+i)(3-i) = 6 - 2i + 3i - i^2 = 6 + i + 1 = 7 + i
    for val in result.data.iter() {
        assert_eq!(*val, Complex::new(7.0, 1.0));
    }
}

/// Test divide operation
#[test]
fn test_divide() {
    let shape = (3, 3, 3);
    let a = WaveArray::from_scalar(shape, Complex::new(10.0, 0.0));
    let b = WaveArray::from_scalar(shape, Complex::new(2.0, 0.0));

    let result = divide(&a, &b);

    for val in result.data.iter() {
        assert_eq!(*val, Complex::new(5.0, 0.0));
    }
}

/// Test lerp operation
#[test]
fn test_lerp() {
    let shape = (3, 3, 3);
    let a = WaveArray::from_scalar(shape, Complex::new(0.0, 0.0));
    let b = WaveArray::from_scalar(shape, Complex::new(10.0, 0.0));
    let weight = WaveArray::from_scalar(shape, Complex::new(0.3, 0.0));
    let mut output = WaveArray::zeros(shape);

    lerp(&a, &b, &weight, &mut output);

    for val in output.data.iter() {
        assert_relative_eq!(val.re, 3.0, epsilon = 1e-10);
    }
}

/// Test copy operation
#[test]
fn test_copy() {
    let shape = (3, 3, 3);
    let source = random_array(shape);
    let mut dest = WaveArray::zeros(shape);

    copy(&source, &mut dest);

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                assert_eq!(source.data[[i, j, k]], dest.data[[i, j, k]]);
            }
        }
    }
}

/// Test arithmetic operations on arrays
#[test]
fn test_arithmetic_operations() {
    let shape = (3, 3, 3);
    let a = WaveArray::from_scalar(shape, Complex::new(5.0, 2.0));
    let b = WaveArray::from_scalar(shape, Complex::new(3.0, 1.0));

    // Test addition
    let sum = a.clone() + b.clone();
    for val in sum.data.iter() {
        assert_eq!(*val, Complex::new(8.0, 3.0));
    }

    // Test subtraction
    let diff = a.clone() - b.clone();
    for val in diff.data.iter() {
        assert_eq!(*val, Complex::new(2.0, 1.0));
    }

    // Test scalar multiplication
    let scaled = a.clone() * Complex::new(2.0, 0.0);
    for val in scaled.data.iter() {
        assert_eq!(*val, Complex::new(10.0, 4.0));
    }

    // Test in-place operations
    let mut c = a.clone();
    c += b.clone();
    for val in c.data.iter() {
        assert_eq!(*val, Complex::new(8.0, 3.0));
    }

    let mut d = a.clone();
    d -= b.clone();
    for val in d.data.iter() {
        assert_eq!(*val, Complex::new(2.0, 1.0));
    }

    let mut e = a.clone();
    e *= Complex::new(2.0, 0.0);
    for val in e.data.iter() {
        assert_eq!(*val, Complex::new(10.0, 4.0));
    }
}

/// Test norm and inner product calculations
#[test]
fn test_norm_and_inner_product() {
    let shape = (2, 2, 2);
    let a = WaveArray::from_scalar(shape, Complex::new(1.0, 1.0));

    // Test norm squared
    // Each element has norm^2 = 1^2 + 1^2 = 2
    // Total = 8 elements * 2 = 16
    let norm_sq = a.norm_squared();
    assert_relative_eq!(norm_sq, 16.0, epsilon = 1e-10);

    // Test inner product
    let b = WaveArray::from_scalar(shape, Complex::new(2.0, -1.0));
    // Inner product = sum of conj(a) * b
    // conj(1+i) * (2-i) = (1-i)(2-i) = 2 - i - 2i + i^2 = 2 - 3i - 1 = 1 - 3i
    // Total = 8 * (1 - 3i)
    let inner = a.inner_product(&b);
    assert_relative_eq!(inner.re, 8.0, epsilon = 1e-10);
    assert_relative_eq!(inner.im, -24.0, epsilon = 1e-10);
}

/// Test FFT operations
#[test]
fn test_fft_operations() {
    let shape = (4, 4, 4);
    let input = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
    let mut fft_result = WaveArray::zeros(shape);
    let mut ifft_result = WaveArray::zeros(shape);

    // Forward FFT
    fft_3d(&input, &mut fft_result);

    // The FFT of a constant should have all energy in the DC component
    assert!(fft_result.data[[0, 0, 0]].norm() > 0.0);

    // Inverse FFT should recover the original
    ifft_3d(&fft_result, &mut ifft_result);

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                assert_relative_eq!(
                    ifft_result.data[[i, j, k]].re,
                    input.data[[i, j, k]].re,
                    epsilon = 1e-10
                );
                assert_relative_eq!(
                    ifft_result.data[[i, j, k]].im,
                    input.data[[i, j, k]].im,
                    epsilon = 1e-10
                );
            }
        }
    }
}

/// Test block array operations
#[test]
fn test_block_array() {
    let shape = (8, 8, 8);
    let data = random_array(shape);

    // Create a block array with 2x2x2 blocks
    let block_array = BlockArray::from_array(data.clone(), [2, 2, 2]);

    // Check that we have 8 blocks (2x2x2)
    assert_eq!(block_array.n_blocks, [2, 2, 2]);

    // Check shape preservation
    assert_eq!(block_array.shape, shape);

    // Test gathering back the data
    let gathered = block_array.gather();
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                assert_eq!(data.data[[i, j, k]], gathered.data[[i, j, k]]);
            }
        }
    }
}

/// Test sparse array operations
#[test]
fn test_sparse_array() {
    let shape = (10, 10, 10);

    // Create a sparse array with a single point
    let value = Complex::new(5.0, -3.0);
    let position = [3, 4, 5];
    // Create a small data block (1x1x1) for the point source
    let point_data = WaveArray::from_scalar((1, 1, 1), value);
    let sparse = SparseArray::new(vec![point_data], vec![position], shape);

    // Convert to dense and check
    let dense = sparse.to_dense();

    // Check that only the specified position has the value
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                if [i, j, k] == position {
                    assert_eq!(dense.data[[i, j, k]], value);
                } else {
                    assert_eq!(dense.data[[i, j, k]], Complex::new(0.0, 0.0));
                }
            }
        }
    }

    // Test that sparse array has correct number of non-zero blocks
    assert_eq!(sparse.nnz(), 1);

    // Test that the shape is preserved
    assert_eq!(sparse.shape, shape);
}

/// Test edge cases and error handling
#[test]
fn test_edge_cases() {
    // Test empty array
    let empty = WaveArray::<Complex64>::zeros((0, 0, 0));
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());

    // Test single element
    let single = WaveArray::from_scalar((1, 1, 1), Complex::new(42.0, 0.0));
    assert_eq!(single.len(), 1);
    assert_eq!(single.data[[0, 0, 0]], Complex::new(42.0, 0.0));

    // Test large array allocation (just check it doesn't panic)
    let large = WaveArray::<Complex64>::zeros((100, 100, 100));
    assert_eq!(large.len(), 1_000_000);
}

/// Test array slicing operations
#[test]
fn test_array_slicing() {
    use wavesim::engine::array::ArraySlice;

    let shape = (10, 10, 10);
    let mut array = WaveArray::zeros(shape);

    // Fill with identifiable pattern
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                array.data[[i, j, k]] = Complex::new((i * 100 + j * 10 + k) as f64, 0.0);
            }
        }
    }

    // Test slicing
    let slice = array.slice([2, 2, 2], [5, 5, 5]);
    assert_eq!(slice.shape_tuple(), (3, 3, 3));

    // Check values
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let expected = Complex::new(((i + 2) * 100 + (j + 2) * 10 + (k + 2)) as f64, 0.0);
                assert_eq!(slice.data[[i, j, k]], expected);
            }
        }
    }

    // Test edges extraction
    let widths = [[1, 1], [1, 1], [1, 1]];
    let edges = array.edges(&widths);

    // Should have 6 edges (2 per dimension)
    assert_eq!(edges.len(), 6);
}
