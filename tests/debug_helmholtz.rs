use num_complex::Complex;
use wavesim::domain::domain_trait::Domain;
use wavesim::domain::helmholtz::HelmholtzDomain;
use wavesim::engine::array::{Complex64, WaveArray};

#[test]
fn test_helmholtz_operators() {
    println!("\n=== Testing Helmholtz operators ===\n");

    // Create a simple homogeneous medium
    let permittivity = WaveArray::from_scalar((5, 5, 5), Complex::new(1.0, 0.0));
    println!("Permittivity norm: {}", permittivity.norm_squared());

    // Create domain with non-periodic boundaries first
    let domain = HelmholtzDomain::new(
        permittivity.clone(),
        0.1,                      // pixel_size
        0.5,                      // wavelength
        [false, false, false],    // non-periodic
        [[1, 1], [1, 1], [1, 1]], // with boundaries
    );

    println!("Domain scale factor: {:?}", domain.scale());
    println!("Domain k02: {:?}", domain.k02);
    println!("Domain shift: {:?}", domain.shift);

    // Test medium operator
    let x = WaveArray::from_scalar((5, 5, 5), Complex::new(1.0, 0.0));
    let mut out_medium = WaveArray::zeros((5, 5, 5));
    domain.medium(&x, &mut out_medium);
    println!(
        "\nMedium operator output norm: {}",
        out_medium.norm_squared()
    );
    println!("First element: {:?}", out_medium.data[[0, 0, 0]]);

    // Test propagator
    let mut out_prop = WaveArray::zeros((5, 5, 5));
    domain.propagator(&x, &mut out_prop);
    println!("\nPropagator output norm: {}", out_prop.norm_squared());
    println!("First element: {:?}", out_prop.data[[0, 0, 0]]);

    // Test inverse propagator
    let mut out_inv_prop = WaveArray::zeros((5, 5, 5));
    domain.inverse_propagator(&x, &mut out_inv_prop);
    println!(
        "\nInverse propagator output norm: {}",
        out_inv_prop.norm_squared()
    );
    println!("First element: {:?}", out_inv_prop.data[[0, 0, 0]]);

    // Check B_scat
    println!("\nB_scat norm: {}", domain.b_scat.norm_squared());
    println!("B_scat first element: {:?}", domain.b_scat.data[[0, 0, 0]]);

    // Now test with periodic boundaries
    println!("\n=== Testing with periodic boundaries ===\n");

    let domain_periodic = HelmholtzDomain::new(
        permittivity,
        0.1,
        0.5,
        [true, true, true],       // periodic
        [[0, 0], [0, 0], [0, 0]], // no boundaries
    );

    let mut out_medium_p = WaveArray::zeros((5, 5, 5));
    domain_periodic.medium(&x, &mut out_medium_p);
    println!(
        "Medium operator output norm (periodic): {}",
        out_medium_p.norm_squared()
    );

    let mut out_prop_p = WaveArray::zeros((5, 5, 5));
    domain_periodic.propagator(&x, &mut out_prop_p);
    println!(
        "Propagator output norm (periodic): {}",
        out_prop_p.norm_squared()
    );
}
