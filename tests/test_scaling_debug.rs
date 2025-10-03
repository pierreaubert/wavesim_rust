//! Debug test for scaling factor calculation

use num_complex::Complex;
use wavesim::domain::helmholtz::HelmholtzDomain;
use wavesim::engine::array::WaveArray;

#[test]
fn test_scaling_for_small_perturbation() {
    println!("\n=== Testing scaling for small perturbations ===");

    // Test 1: Homogeneous medium
    let shape = (4, 4, 4);
    let permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

    let domain = HelmholtzDomain::new(
        permittivity,
        0.25, // pixel_size
        1.0,  // wavelength
        [true, true, true],
        [[0, 0], [0, 0], [0, 0]],
    );

    println!("Homogeneous medium (ε = 1.0):");
    println!("  k0^2 = {:?}", domain.k02);
    println!("  Scale factor = {:?}", domain.scale_factor);
    println!("  Shift = {:?}", domain.shift);
    println!("  B_scat[0,0,0] = {:?}", domain.b_scat.data[[0, 0, 0]]);

    // Test 2: Small perturbation
    let mut permittivity2 = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
    permittivity2.data[[2, 2, 2]] = Complex::new(1.05, 0.0);

    let domain2 = HelmholtzDomain::new(
        permittivity2,
        0.25, // pixel_size
        1.0,  // wavelength
        [true, true, true],
        [[0, 0], [0, 0], [0, 0]],
    );

    println!("\nSmall perturbation (ε = 1.0 except one point at 1.05):");
    println!("  Scale factor = {:?}", domain2.scale_factor);
    println!("  Shift = {:?}", domain2.shift);
    println!("  B_scat[0,0,0] = {:?}", domain2.b_scat.data[[0, 0, 0]]);
    println!("  B_scat[2,2,2] = {:?}", domain2.b_scat.data[[2, 2, 2]]);

    // Test 3: Larger variation
    let mut permittivity3 = WaveArray::from_scalar(shape, Complex::new(1.5, 0.0));
    permittivity3.data[[2, 2, 2]] = Complex::new(2.0, 0.0);

    let domain3 = HelmholtzDomain::new(
        permittivity3,
        0.25, // pixel_size
        1.0,  // wavelength
        [true, true, true],
        [[0, 0], [0, 0], [0, 0]],
    );

    println!("\nLarger variation (ε = 1.5 except one point at 2.0):");
    println!("  Scale factor = {:?}", domain3.scale_factor);
    println!("  Shift = {:?}", domain3.shift);
    println!("  B_scat[0,0,0] = {:?}", domain3.b_scat.data[[0, 0, 0]]);
    println!("  B_scat[2,2,2] = {:?}", domain3.b_scat.data[[2, 2, 2]]);

    // The scale factor should be reasonable for convergence
    assert!(
        domain2.scale_factor.norm() > 0.0,
        "Scale factor should be non-zero"
    );
    assert!(
        domain2.scale_factor.norm() < 1.0,
        "Scale factor should be bounded"
    );
}
