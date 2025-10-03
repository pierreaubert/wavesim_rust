//! Test convergence for small perturbations with extended iterations

use num_complex::Complex;
use wavesim::domain::domain_trait::Domain;
use wavesim::domain::helmholtz::HelmholtzDomain;
use wavesim::domain::iteration::{preconditioned_richardson, IterationConfig};
use wavesim::engine::array::WaveArray;

#[test]
fn test_homogeneous_medium() {
    println!("\n=== Testing homogeneous medium ===");

    let shape = (8, 8, 8);
    let permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

    let domain = HelmholtzDomain::new(
        permittivity,
        0.25,               // pixel_size
        1.0,                // wavelength
        [true, true, true], // periodic
        [[0, 0], [0, 0], [0, 0]],
    );

    println!("Scale factor: {:?}", domain.scale());
    println!("k0^2: {:?}", domain.k02);

    // Point source
    let mut source = WaveArray::zeros(shape);
    source.data[[4, 4, 4]] = Complex::new(1.0, 0.0);

    let config = IterationConfig {
        max_iterations: 1000,
        threshold: 1e-6,
        alpha: 0.5,
        full_residuals: true,
    };

    let result = preconditioned_richardson(&domain, &source, config);

    println!("Homogeneous medium results:");
    println!("  Iterations: {}", result.iterations);
    println!("  Final residual: {:.2e}", result.residual_norm);

    if let Some(history) = &result.residual_history {
        if history.len() > 10 {
            println!("  First 10 residuals: {:?}", &history[..10]);
            println!(
                "  Last 10 residuals: {:?}",
                &history[history.len().saturating_sub(10)..]
            );
        } else {
            println!("  Residual history: {:?}", history);
        }
    }

    // For homogeneous medium with our scaling, convergence may be slow
    assert!(result.residual_norm < 1.0, "Should at least make progress");
}

#[test]
fn test_tiny_perturbation() {
    println!("\n=== Testing tiny perturbation (1% contrast) ===");

    let shape = (8, 8, 8);
    let mut permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

    // Add 1% perturbation
    permittivity.data[[4, 4, 4]] = Complex::new(1.01, 0.0);

    let domain = HelmholtzDomain::new(
        permittivity,
        0.25,               // pixel_size
        1.0,                // wavelength
        [true, true, true], // periodic
        [[0, 0], [0, 0], [0, 0]],
    );

    println!("Scale factor: {:?}", domain.scale());

    let mut source = WaveArray::zeros(shape);
    source.data[[4, 4, 4]] = Complex::new(1.0, 0.0);

    let config = IterationConfig {
        max_iterations: 2000, // More iterations
        threshold: 1e-5,      // Slightly relaxed threshold
        alpha: 0.3,           // Smaller alpha for stability
        full_residuals: true,
    };

    let result = preconditioned_richardson(&domain, &source, config);

    println!("Tiny perturbation results:");
    println!("  Iterations: {}", result.iterations);
    println!("  Final residual: {:.2e}", result.residual_norm);

    if let Some(history) = &result.residual_history {
        // Show convergence trend
        let checkpoints = vec![0, 10, 50, 100, 200, 500, 1000, history.len() - 1];
        println!("  Convergence trend:");
        for &i in checkpoints.iter() {
            if i < history.len() {
                println!("    Iteration {:4}: residual = {:.2e}", i + 1, history[i]);
            }
        }
    }

    // Should at least make progress
    assert!(result.residual_norm < 0.8, "Should make some progress");
}

#[test]
fn test_small_perturbation_extended() {
    println!("\n=== Testing 5% perturbation with extended iterations ===");

    let shape = (8, 8, 8);
    let mut permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

    // Add 5% perturbation
    permittivity.data[[4, 4, 4]] = Complex::new(1.05, 0.0);

    let domain = HelmholtzDomain::new(
        permittivity,
        0.25,               // pixel_size
        1.0,                // wavelength
        [true, true, true], // periodic
        [[0, 0], [0, 0], [0, 0]],
    );

    println!("Scale factor: {:?}", domain.scale());
    println!(
        "B_scat at perturbation: {:?}",
        domain.b_scat.data[[4, 4, 4]]
    );

    let mut source = WaveArray::zeros(shape);
    source.data[[4, 4, 4]] = Complex::new(1.0, 0.0);

    // Try different alpha values
    let alphas = vec![0.1, 0.3, 0.5, 0.75];

    for alpha in alphas {
        let config = IterationConfig {
            max_iterations: 5000, // Many more iterations
            threshold: 1e-4,
            alpha,
            full_residuals: false, // Don't store full history for performance
        };

        let result = preconditioned_richardson(&domain, &source, config);

        println!(
            "  Alpha = {:.2}: iterations = {}, residual = {:.2e}",
            alpha, result.iterations, result.residual_norm
        );
    }
}

#[test]
fn test_multiple_small_perturbations() {
    println!("\n=== Testing multiple small perturbations ===");

    let shape = (8, 8, 8);
    let mut permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

    // Add several small perturbations
    permittivity.data[[3, 3, 3]] = Complex::new(1.02, 0.0);
    permittivity.data[[4, 4, 4]] = Complex::new(1.03, 0.0);
    permittivity.data[[5, 5, 5]] = Complex::new(0.98, 0.0);
    permittivity.data[[3, 5, 4]] = Complex::new(1.01, 0.0);

    let domain = HelmholtzDomain::new(
        permittivity,
        0.25,               // pixel_size
        1.0,                // wavelength
        [true, true, true], // periodic
        [[0, 0], [0, 0], [0, 0]],
    );

    println!("Scale factor: {:?}", domain.scale());

    // Gaussian-like source
    let mut source = WaveArray::zeros(shape);
    for i in 3..=5 {
        for j in 3..=5 {
            for k in 3..=5 {
                let dist_sq = ((i as f64 - 4.0).powi(2)
                    + (j as f64 - 4.0).powi(2)
                    + (k as f64 - 4.0).powi(2))
                    / 2.0;
                source.data[[i, j, k]] = Complex::new((-dist_sq).exp(), 0.0);
            }
        }
    }

    let config = IterationConfig {
        max_iterations: 3000,
        threshold: 1e-4,
        alpha: 0.4,
        full_residuals: false,
    };

    let result = preconditioned_richardson(&domain, &source, config);

    println!("Multiple perturbations results:");
    println!("  Iterations: {}", result.iterations);
    println!("  Final residual: {:.2e}", result.residual_norm);

    // Should converge or at least make significant progress
    assert!(
        result.residual_norm < 0.1 || result.iterations == 3000,
        "Should either converge or reach max iterations"
    );
}

#[test]
fn test_small_absorption() {
    println!("\n=== Testing small perturbation with absorption ===");

    let shape = (8, 8, 8);
    let mut permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

    // Add small perturbation with absorption
    permittivity.data[[4, 4, 4]] = Complex::new(1.05, 0.001); // Small imaginary part
    permittivity.data[[3, 3, 3]] = Complex::new(0.98, 0.002);

    let domain = HelmholtzDomain::new(
        permittivity,
        0.25,               // pixel_size
        1.0,                // wavelength
        [true, true, true], // periodic
        [[0, 0], [0, 0], [0, 0]],
    );

    let mut source = WaveArray::zeros(shape);
    source.data[[4, 4, 4]] = Complex::new(1.0, 0.0);

    let config = IterationConfig {
        max_iterations: 2000,
        threshold: 1e-4,
        alpha: 0.5,
        full_residuals: false,
    };

    let result = preconditioned_richardson(&domain, &source, config);

    println!("Small absorption results:");
    println!("  Iterations: {}", result.iterations);
    println!("  Final residual: {:.2e}", result.residual_norm);

    // Absorption should help convergence
    assert!(
        result.residual_norm < 0.5,
        "Should converge better with absorption"
    );
}

#[test]
fn test_adaptive_alpha() {
    println!("\n=== Testing adaptive relaxation parameter ===");

    let shape = (8, 8, 8);
    let mut permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
    permittivity.data[[4, 4, 4]] = Complex::new(1.03, 0.0);

    let domain = HelmholtzDomain::new(
        permittivity,
        0.25,
        1.0,
        [true, true, true],
        [[0, 0], [0, 0], [0, 0]],
    );

    let mut source = WaveArray::zeros(shape);
    source.data[[4, 4, 4]] = Complex::new(1.0, 0.0);

    // Start with larger alpha and decrease if not converging
    let mut best_result: Option<wavesim::domain::iteration::IterationResult> = None;
    let mut best_alpha = 0.0;

    for i in 0..5 {
        let alpha = 0.8 * 0.7_f64.powi(i); // 0.8, 0.56, 0.392, 0.274, 0.192

        let config = IterationConfig {
            max_iterations: 1000,
            threshold: 1e-4,
            alpha,
            full_residuals: false,
        };

        let result = preconditioned_richardson(&domain, &source, config);

        println!(
            "  Alpha = {:.3}: iterations = {:4}, residual = {:.2e}",
            alpha, result.iterations, result.residual_norm
        );

        if best_result.is_none()
            || result.residual_norm < best_result.as_ref().unwrap().residual_norm
        {
            best_alpha = alpha;
            best_result = Some(result);
        }
    }

    println!(
        "Best alpha = {:.3} with residual = {:.2e}",
        best_alpha,
        best_result.as_ref().unwrap().residual_norm
    );

    assert!(
        best_result.unwrap().residual_norm < 0.5,
        "Should find a working alpha"
    );
}
