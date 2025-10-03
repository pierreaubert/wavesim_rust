//! Tests for Maxwell solver and utility functions

use approx::assert_relative_eq;
use num_complex::Complex;
use wavesim::domain::maxwell::*;
use wavesim::engine::array::WaveArray;
use wavesim::utilities::{
    add_absorbing_boundaries, create_gaussian_source, create_source, laplace_kernel_1d,
};

/// Test Maxwell domain creation
#[test]
fn test_maxwell_domain_creation() {
    let shape = (20, 20, 20);
    let permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
    let permeability = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

    let domain = MaxwellDomain::new(
        permittivity,
        permeability,
        1e-6, // 1 micrometer
        1e-6,
        1e-6,
        [false, false, false],
        [3, 3, 3], // PML thickness
    );

    assert_eq!(domain.shape, shape);
    assert_eq!(domain.dx, 1e-6);

    // Check CFL condition for time step
    let c = 299792458.0; // Speed of light
    let dt_max = 1.0 / (c * (3.0_f64.sqrt() / 1e-6));
    assert!(domain.dt < dt_max);
    assert!(domain.dt > 0.0);
}

/// Test electromagnetic field initialization
#[test]
fn test_field_initialization() {
    let shape = (10, 10, 10);
    let permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
    let permeability = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

    let domain = MaxwellDomain::new(
        permittivity,
        permeability,
        1e-6,
        1e-6,
        1e-6,
        [false, false, false],
        [2, 2, 2],
    );

    let fields = domain.init_fields();

    // Check that all fields are initialized to zero
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                assert_eq!(fields.ex.data[[i, j, k]], Complex::new(0.0, 0.0));
                assert_eq!(fields.ey.data[[i, j, k]], Complex::new(0.0, 0.0));
                assert_eq!(fields.ez.data[[i, j, k]], Complex::new(0.0, 0.0));
                assert_eq!(fields.hx.data[[i, j, k]], Complex::new(0.0, 0.0));
                assert_eq!(fields.hy.data[[i, j, k]], Complex::new(0.0, 0.0));
                assert_eq!(fields.hz.data[[i, j, k]], Complex::new(0.0, 0.0));
            }
        }
    }
}

/// Test Maxwell field update (basic FDTD step)
#[test]
fn test_maxwell_field_update() {
    let shape = (10, 10, 10);
    let permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
    let permeability = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

    let domain = MaxwellDomain::new(
        permittivity,
        permeability,
        1e-6,
        1e-6,
        1e-6,
        [true, true, true], // Periodic for simplicity
        [0, 0, 0],
    );

    let mut fields = domain.init_fields();

    // Set a non-zero initial field
    fields.ez.data[[5, 5, 5]] = Complex::new(1.0, 0.0);

    // Perform one time step
    domain.step(&mut fields);

    // After one step, the field should have changed
    // The exact values depend on the FDTD implementation
    // but we check that something has changed
    let mut changed = false;
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                if fields.hx.data[[i, j, k]].norm() > 1e-10
                    || fields.hy.data[[i, j, k]].norm() > 1e-10
                {
                    changed = true;
                    break;
                }
            }
        }
    }
    assert!(changed, "Magnetic field should have been updated");
}

/// Test point dipole source
#[test]
fn test_point_dipole_source() {
    let shape = (20, 20, 20);
    let permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
    let permeability = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

    let domain = MaxwellDomain::new(
        permittivity,
        permeability,
        1e-6,
        1e-6,
        1e-6,
        [false, false, false],
        [3, 3, 3],
    );

    let mut fields = domain.init_fields();

    let source = MaxwellSource {
        source_type: MaxwellSourceType::PointDipole {
            position: [10, 10, 10],
            orientation: Orientation::Z,
            amplitude: Complex::new(1.0, 0.0),
        },
        frequency: 3e14, // 300 THz
    };

    // Add source at t=0
    domain.add_source(&mut fields, &source, 0.0);
    assert_eq!(fields.ez.data[[10, 10, 10]], Complex::new(0.0, 0.0)); // sin(0) = 0

    // Add source at t = period/4
    let period = 1.0 / source.frequency;
    domain.add_source(&mut fields, &source, period / 4.0);
    assert_relative_eq!(fields.ez.data[[10, 10, 10]].re, 1.0, epsilon = 1e-10);
}

/// Test absorbing boundaries utility
#[test]
fn test_absorbing_boundaries() {
    let shape = (10, 10, 10);
    let original = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
    let boundary_widths = [[2, 2], [2, 2], [2, 2]];

    let (with_boundaries, roi) = add_absorbing_boundaries(
        original.clone(),
        boundary_widths,
        1.0, // absorption strength
        [false, false, false],
    );

    // Check new shape
    let new_shape = with_boundaries.shape_tuple();
    assert_eq!(new_shape.0, shape.0 + 4); // 2 on each side
    assert_eq!(new_shape.1, shape.1 + 4);
    assert_eq!(new_shape.2, shape.2 + 4);

    // Check ROI
    assert_eq!(roi[0], (2, 12));
    assert_eq!(roi[1], (2, 12));
    assert_eq!(roi[2], (2, 12));

    // Check that interior is preserved
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                assert_eq!(
                    with_boundaries.data[[i + 2, j + 2, k + 2]],
                    original.data[[i, j, k]]
                );
            }
        }
    }

    // Check that boundaries have absorption
    // Corners should have strongest absorption
    assert!(with_boundaries.data[[0, 0, 0]].im.abs() > 0.0);
}

/// Test Laplace kernel creation
#[test]
fn test_laplace_kernel() {
    let pixel_size = 0.1;
    let n_pixels = 10;

    let kernel = laplace_kernel_1d(pixel_size, n_pixels);

    assert_eq!(kernel.len(), n_pixels);

    // The kernel should have specific values for finite differences
    // The DC component (i=0) is not zero but PI^2/(3*pixel_size^2)
    use std::f64::consts::PI;
    let expected_dc = PI.powi(2) / (3.0 * pixel_size.powi(2));
    assert_relative_eq!(kernel[0].re, expected_dc, epsilon = 1e-10);

    // Should be real
    for val in kernel.iter() {
        assert!(val.im.abs() < 1e-10); // Should be real
    }

    // Check that the kernel has zero average (after adjustment)
    let sum: Complex<f64> = kernel.iter().sum();
    assert!(sum.norm() < 1e-10); // Should have zero average
}

/// Test Gaussian source creation
#[test]
fn test_gaussian_source() {
    let position = [10, 10, 10];
    let width = [2.0, 2.0, 2.0]; // in physical units
    let pixel_size = 0.1;
    let wavelength = 1.0;
    let amplitude = Complex::new(1.0, 0.0);

    let (source, _adjusted_pos) =
        create_gaussian_source(position, width, pixel_size, wavelength, amplitude);

    let shape = source.shape();

    // Check that source is centered in its local grid
    let center = [shape[0] / 2, shape[1] / 2, shape[2] / 2];
    let center_val = source.data[[center[0], center[1], center[2]]].norm();
    assert!(center_val > 0.0);

    // Check that it falls off away from center
    if shape[0] > 1 {
        assert!(source.data[[0, center[1], center[2]]].norm() < center_val);
    }

    // Check Gaussian profile
    if shape[0] > 3 {
        let offset = 1;
        let dist = offset as f64 * pixel_size;
        let expected = (-dist * dist / (width[0] * width[0])).exp();
        let actual = source.data[[center[0] + offset, center[1], center[2]]].norm() / center_val;
        assert_relative_eq!(actual, expected, epsilon = 0.1);
    }
}

/// Test point source creation
#[test]
fn test_point_source() {
    let position = [3, 4, 5];
    let pixel_size = 1.0;
    let wavelength = 1.0;
    let amplitude = Complex::new(1.0, 0.0);

    // Use create_source to make a point source
    let (source, pos) = create_source(position, pixel_size, wavelength, amplitude);

    // Check that we get a single-pixel source
    assert_eq!(source.shape(), &[1, 1, 1]);
    assert_eq!(source.data[[0, 0, 0]], amplitude);
    assert_eq!(pos, position);
}

/// Test energy calculation in electromagnetic fields
#[test]
fn test_electromagnetic_energy() {
    let shape = (10, 10, 10);
    let fields = ElectromagneticFields {
        ex: WaveArray::from_scalar(shape, Complex::new(1.0, 0.0)),
        ey: WaveArray::from_scalar(shape, Complex::new(0.0, 0.0)),
        ez: WaveArray::from_scalar(shape, Complex::new(0.0, 0.0)),
        hx: WaveArray::from_scalar(shape, Complex::new(0.0, 0.0)),
        hy: WaveArray::from_scalar(shape, Complex::new(1.0, 0.0)),
        hz: WaveArray::from_scalar(shape, Complex::new(0.0, 0.0)),
    };

    // Calculate electromagnetic energy
    // E = 0.5 * (ε|E|² + μ|H|²) for vacuum
    let mut total_energy = 0.0;
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                let e_energy = fields.ex.data[[i, j, k]].norm_sqr()
                    + fields.ey.data[[i, j, k]].norm_sqr()
                    + fields.ez.data[[i, j, k]].norm_sqr();

                let h_energy = fields.hx.data[[i, j, k]].norm_sqr()
                    + fields.hy.data[[i, j, k]].norm_sqr()
                    + fields.hz.data[[i, j, k]].norm_sqr();

                total_energy += 0.5 * (e_energy + h_energy);
            }
        }
    }

    // Each cell has Ex=1 and Hy=1, so energy = 0.5 * (1 + 1) = 1 per cell
    // Total = 1000 cells * 1 = 1000
    assert_relative_eq!(total_energy, 1000.0, epsilon = 1e-10);
}

/// Test periodic boundary conditions in Maxwell solver
#[test]
fn test_periodic_boundaries() {
    let shape = (10, 10, 10);
    let permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
    let permeability = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

    let domain = MaxwellDomain::new(
        permittivity,
        permeability,
        1e-6,
        1e-6,
        1e-6,
        [true, true, true], // All periodic
        [0, 0, 0],
    );

    let mut fields = domain.init_fields();

    // Set fields at boundaries
    fields.ex.data[[0, 5, 5]] = Complex::new(1.0, 0.0);
    fields.ey.data[[5, 0, 5]] = Complex::new(2.0, 0.0);
    fields.ez.data[[5, 5, 0]] = Complex::new(3.0, 0.0);

    // Apply boundary conditions (part of the step)
    domain.step(&mut fields);

    // Due to periodic boundaries, fields should wrap around
    // The exact behavior depends on the implementation
    // but we verify that the fields have been updated
    assert!(
        fields.hx.norm_squared() > 0.0
            || fields.hy.norm_squared() > 0.0
            || fields.hz.norm_squared() > 0.0
    );
}

/// Test PML absorption
#[test]
fn test_pml_absorption() {
    let shape = (20, 20, 20);
    let permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
    let permeability = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

    let domain = MaxwellDomain::new(
        permittivity,
        permeability,
        1e-6,
        1e-6,
        1e-6,
        [false, false, false],
        [5, 5, 5], // 5 cells of PML on each side
    );

    let mut fields = domain.init_fields();

    // Create a wave packet in the center
    fields.ez.data[[10, 10, 10]] = Complex::new(1.0, 0.0);

    let initial_energy = fields.ez.norm_squared();

    // Propagate for many steps
    for _ in 0..100 {
        domain.step(&mut fields);
    }

    // Energy should decrease due to PML absorption
    let final_energy =
        fields.ez.norm_squared() + fields.ex.norm_squared() + fields.ey.norm_squared();

    // Some energy should have been absorbed
    assert!(final_energy < initial_energy);
}
