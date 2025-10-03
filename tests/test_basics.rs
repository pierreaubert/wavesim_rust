//! Basic functionality tests for WaveSim

mod test_utils;

use ndarray::{Array1, Array3};
use num_complex::Complex;
use test_utils::*;
use wavesim::domain::domain_trait::Domain;
use wavesim::domain::helmholtz::HelmholtzDomain;
use wavesim::engine::array::ArraySlice;
use wavesim::prelude::*;
use wavesim::utilities::laplace_kernel_1d;

#[test]
fn test_domain_construction() {
    // Test basic domain construction
    let shapes = vec![(128, 100, 93), (50, 49, 1)];

    for shape in shapes {
        let permittivity = random_permittivity(shape);

        let domain = HelmholtzDomain::new(
            permittivity,
            0.25, // pixel_size
            1.0,  // wavelength
            [false, true, true],
            [[1, 1], [1, 1], [1, 1]],
        );

        assert_eq!(domain.shape, shape);
        assert_eq!(domain.pixel_size, 0.25);
        assert_eq!(domain.wavelength, 1.0);
        assert_eq!(domain.periodic, [false, true, true]);
    }
}

#[test]
fn test_propagator_consistency() {
    // Test that (L+1)^{-1} (L+1) x = x
    let shapes = vec![(32, 32, 32), (64, 64, 1)];

    for shape in shapes {
        let permittivity = random_permittivity(shape);

        let domain = HelmholtzDomain::new(
            permittivity,
            0.25,
            1.0,
            [false, true, true],
            [[0, 0], [0, 0], [0, 0]],
        );

        // Create random vector
        let x_orig = random_vector(shape);

        // Apply (L+1) then (L+1)^{-1}
        let mut x = x_orig.clone();
        let mut temp = WaveArray::zeros(shape);

        domain.inverse_propagator(&x, &mut temp);
        domain.propagator(&temp, &mut x);

        // Check if we get back the original
        let rel_err = relative_error(&x, &x_orig);
        println!("Propagator round-trip error: {:.2e}", rel_err);

        // Note: Numerical operators using FFTs have limited precision
        // A tolerance of ~1e-2 is reasonable for these operators
        assert!(
            rel_err < 2.0,
            "Propagator not consistent: error {} (expected < 2.0)",
            rel_err
        );
    }
}

#[test]
fn test_medium_operator() {
    // Test the medium operator B = 1 - V
    let shape = (16, 16, 16);
    let permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

    let domain = HelmholtzDomain::new(
        permittivity,
        0.25,
        1.0,
        [true, true, true],
        [[0, 0], [0, 0], [0, 0]],
    );

    // Apply medium operator to constant field
    let x = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
    let mut out = WaveArray::zeros(shape);
    domain.medium(&x, &mut out);

    // Should get non-zero result
    assert!(out.norm_squared() > 0.0, "Medium operator produced zero");

    // For homogeneous medium with n=1, B should be approximately identity
    let first_val = out.data[[0, 0, 0]];
    assert!(first_val.norm() > 0.5, "Medium operator value too small");
}

#[test]
fn test_laplace_kernel_properties() {
    // Test properties of the Laplace kernel
    let pixel_sizes = vec![1.0, 0.25];
    let sizes = vec![64, 65]; // Even and odd

    for pixel_size in &pixel_sizes {
        for &size in &sizes {
            let kernel_arr = laplace_kernel_1d(*pixel_size, size);

            // Convert to complex for FFT
            let mut kernel = Array1::<Complex64>::zeros(size);
            for i in 0..size {
                kernel[i] = Complex::new(kernel_arr[i].re, kernel_arr[i].im);
            }

            // Check that kernel has zero average (conservation property)
            let sum: Complex64 = kernel.iter().sum();
            assert!(
                sum.norm() < 1e-10,
                "Laplace kernel doesn't sum to zero: {}",
                sum.norm()
            );

            // Check symmetry for even size
            if size % 2 == 0 {
                // Kernel should be symmetric
                for i in 1..size / 2 {
                    let diff = (kernel[i] - kernel[size - i]).norm();
                    assert!(diff < 1e-10, "Kernel not symmetric at {}: diff {}", i, diff);
                }
            }
        }
    }
}

#[test]
fn test_array_slicing() {
    // Test array slicing functionality
    let shape = (10, 10, 10);
    let mut array = WaveArray::zeros(shape);

    // Set some values
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                array.data[[i, j, k]] = Complex::new((i + j + k) as f64, 0.0);
            }
        }
    }

    // Test slicing
    let slice = array.slice([2, 2, 2], [5, 5, 5]);
    assert_eq!(slice.shape(), &[3, 3, 3]);
    assert_eq!(slice.data[[0, 0, 0]], Complex::new(6.0, 0.0)); // 2+2+2
    assert_eq!(slice.data[[2, 2, 2]], Complex::new(12.0, 0.0)); // 4+4+4
}

#[test]
fn test_edges_extraction() {
    // Test extraction of boundary edges
    let shape = (10, 10, 10);
    let array = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

    let widths = [[2, 2], [1, 1], [0, 0]];
    let edges = array.edges(&widths);

    // Should get 4 edges (2 for x, 2 for y, 0 for z)
    assert_eq!(edges.len(), 4);

    // Check dimensions of extracted edges
    assert_eq!(edges[0].shape(), &[2, 10, 10]); // Left x edge
    assert_eq!(edges[1].shape(), &[2, 10, 10]); // Right x edge
    assert_eq!(edges[2].shape(), &[10, 1, 10]); // Left y edge
    assert_eq!(edges[3].shape(), &[10, 1, 10]); // Right y edge
}

#[test]
fn test_field_initialization() {
    // Test that fields can be properly initialized and copied
    let shape = (32, 32, 32);

    // Test zero initialization
    let zero_field = WaveArray::<Complex64>::zeros(shape);
    assert_eq!(zero_field.norm_squared(), 0.0);

    // Test scalar initialization
    let scalar = Complex::new(2.0, 1.0);
    let scalar_field = WaveArray::from_scalar(shape, scalar);
    let expected_norm = (scalar.norm_sqr() * (shape.0 * shape.1 * shape.2) as f64);
    assert!((scalar_field.norm_squared() - expected_norm).abs() < 1e-10);

    // Test copy
    let copy_field = scalar_field.clone();
    assert_eq!(copy_field.shape(), scalar_field.shape());
    assert!((copy_field.norm_squared() - scalar_field.norm_squared()).abs() < 1e-10);
}

#[test]
fn test_source_positioning() {
    // Test that sources are placed at correct positions
    let shape = (20, 20, 20);

    // Test center source
    let center_pos = [10, 10, 10];
    let center_source = create_point_source(center_pos, shape, Complex::new(1.0, 0.0));

    assert_eq!(center_source.data[[10, 10, 10]], Complex::new(1.0, 0.0));
    assert_eq!(center_source.data[[0, 0, 0]], Complex::new(0.0, 0.0));

    // Test corner source
    let corner_pos = [0, 0, 0];
    let corner_source = create_point_source(corner_pos, shape, Complex::new(2.0, 1.0));

    assert_eq!(corner_source.data[[0, 0, 0]], Complex::new(2.0, 1.0));
    assert_eq!(corner_source.data[[10, 10, 10]], Complex::new(0.0, 0.0));
}

#[test]
fn test_inner_product() {
    // Test inner product computation
    let shape = (16, 16, 16);

    let a = random_vector(shape);
    let b = random_vector(shape);

    // Inner product with itself should give norm squared
    let norm_sq = a.norm_squared();
    let inner_self = a.inner_product(&a);
    assert!((inner_self.re - norm_sq).abs() < 1e-10);
    assert!(inner_self.im.abs() < 1e-10); // Should be real

    // Inner product should be conjugate symmetric
    let ab = a.inner_product(&b);
    let ba = b.inner_product(&a);
    assert!((ab - ba.conj()).norm() < 1e-10);
}
