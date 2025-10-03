//! Debug test for convergence issues

use num_complex::Complex;
use wavesim::domain::domain_trait::Domain;
use wavesim::domain::helmholtz::HelmholtzDomain;
use wavesim::domain::iteration::{preconditioned_richardson, IterationConfig};
use wavesim::engine::array::{Complex64, WaveArray};

#[test]
fn test_debug_convergence() {
    // Create the simplest possible case - homogeneous medium
    let shape = (4, 4, 4);
    let permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

    let domain = HelmholtzDomain::new(
        permittivity,
        0.25,               // pixel_size
        1.0,                // wavelength
        [true, true, true], // periodic
        [[0, 0], [0, 0], [0, 0]],
    );

    println!("Scale factor: {:?}", domain.scale());
    println!("Shift: {:?}", domain.shift);
    println!("k02: {:?}", domain.k02);

    // Create a simple source
    let mut source = WaveArray::zeros(shape);
    source.data[[2, 2, 2]] = Complex::new(1.0, 0.0);

    // Test medium operator
    let mut b_source = WaveArray::zeros(shape);
    domain.medium(&source, &mut b_source);
    println!("Medium operator norm: {}", b_source.norm_squared().sqrt());

    // Test propagator
    let mut prop_source = WaveArray::zeros(shape);
    domain.propagator(&source, &mut prop_source);
    println!(
        "Propagator result norm: {}",
        prop_source.norm_squared().sqrt()
    );

    // Test inverse propagator
    let mut inv_prop_source = WaveArray::zeros(shape);
    domain.inverse_propagator(&source, &mut inv_prop_source);
    println!(
        "Inverse propagator result norm: {}",
        inv_prop_source.norm_squared().sqrt()
    );

    // Check if (L+1)^{-1} * (L+1) = I
    let mut roundtrip = WaveArray::zeros(shape);
    domain.propagator(&inv_prop_source, &mut roundtrip);

    println!("\nRoundtrip test (should be close to source):");
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                let diff = (roundtrip.data[[i, j, k]] - source.data[[i, j, k]]).norm();
                if diff > 1e-10 {
                    println!(
                        "  [{},{},{}]: source={:?}, roundtrip={:?}, diff={}",
                        i,
                        j,
                        k,
                        source.data[[i, j, k]],
                        roundtrip.data[[i, j, k]],
                        diff
                    );
                }
            }
        }
    }

    // Test B operator values
    println!("\nB_scat values (should be close to 1 for homogeneous):");
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                println!(
                    "  B[{},{},{}] = {:?}",
                    i,
                    j,
                    k,
                    domain.b_scat.data[[i, j, k]]
                );
            }
        }
    }

    // Try a single Richardson iteration
    let config = IterationConfig {
        max_iterations: 10,
        threshold: 1e-6,
        alpha: 0.75,
        full_residuals: true,
    };

    let result = preconditioned_richardson(&domain, &source, config);

    println!("\nRichardson iteration:");
    println!("  Iterations: {}", result.iterations);
    println!("  Final residual: {}", result.residual_norm);
    if let Some(history) = result.residual_history {
        println!("  Residual history: {:?}", history);
    }

    // For homogeneous medium with k0^2 = (2π/1.0)^2 ≈ 39.5
    // The solution should exist and converge
    assert!(
        result.residual_norm < 1.0,
        "Should at least improve from initial"
    );
}

#[test]
fn test_fft_normalization() {
    use wavesim::engine::operations::{fft_3d, ifft_3d};

    let shape = (4, 4, 4);
    let mut input = WaveArray::zeros(shape);

    // Set a single point
    input.data[[1, 1, 1]] = Complex::new(1.0, 0.0);

    let mut fft_result = WaveArray::zeros(shape);
    let mut ifft_result = WaveArray::zeros(shape);

    // Forward and inverse FFT
    fft_3d(&input, &mut fft_result);
    ifft_3d(&fft_result, &mut ifft_result);

    // Check normalization
    println!("FFT normalization test:");
    println!("  Input norm: {}", input.norm_squared().sqrt());
    println!("  FFT result norm: {}", fft_result.norm_squared().sqrt());
    println!("  IFFT result norm: {}", ifft_result.norm_squared().sqrt());

    // Should recover the original
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                let diff = (ifft_result.data[[i, j, k]] - input.data[[i, j, k]]).norm();
                if diff > 1e-10 {
                    println!(
                        "  [{},{},{}]: input={:?}, recovered={:?}, diff={}",
                        i,
                        j,
                        k,
                        input.data[[i, j, k]],
                        ifft_result.data[[i, j, k]],
                        diff
                    );
                }
            }
        }
    }

    // Check exact recovery
    assert!(
        (ifft_result.data[[1, 1, 1]] - Complex::new(1.0, 0.0)).norm() < 1e-10,
        "FFT/IFFT roundtrip should preserve values"
    );
}
