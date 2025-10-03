//! Diagnostic test for small perturbation convergence issues

use num_complex::Complex;
use wavesim::domain::domain_trait::Domain;
use wavesim::domain::helmholtz::HelmholtzDomain;
use wavesim::domain::iteration::{preconditioned_richardson, IterationConfig};
use wavesim::engine::array::WaveArray;

#[test]
fn diagnose_small_perturbation_issue() {
    println!("\n=== Diagnosing Small Perturbation Convergence ===\n");

    let shape = (4, 4, 4);

    // Test different perturbation levels
    let perturbations = vec![0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5];

    for pert in perturbations {
        let mut permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
        if pert > 0.0 {
            permittivity.data[[2, 2, 2]] = Complex::new(1.0 + pert, 0.0);
        }

        let domain = HelmholtzDomain::new(
            permittivity,
            0.25,
            1.0,
            [true, true, true],
            [[0, 0], [0, 0], [0, 0]],
        );

        println!("Perturbation = {:.1}%:", pert * 100.0);
        println!("  Scale factor = {:?}", domain.scale());
        println!("  ||scale|| = {:.4}", domain.scale().norm());

        // Check B_scat values
        let b_uniform = domain.b_scat.data[[0, 0, 0]];
        let b_perturbed = if pert > 0.0 {
            domain.b_scat.data[[2, 2, 2]]
        } else {
            b_uniform
        };

        println!("  B_scat (uniform) = {:?}", b_uniform);
        if pert > 0.0 {
            println!("  B_scat (perturbed) = {:?}", b_perturbed);
            println!(
                "  ||B_scat|| range = [{:.4}, {:.4}]",
                b_uniform.norm(),
                b_perturbed.norm()
            );
        }

        // Test convergence with a simple source
        let mut source = WaveArray::zeros(shape);
        source.data[[2, 2, 2]] = Complex::new(1.0, 0.0);

        // Quick convergence test
        let config = IterationConfig {
            max_iterations: 100,
            threshold: 1e-4,
            alpha: 0.5,
            full_residuals: true,
        };

        let result = preconditioned_richardson(&domain, &source, config);

        println!(
            "  Convergence (100 iter): residual = {:.2e}",
            result.residual_norm
        );

        // Check operator norms
        let mut test_vec = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
        let mut medium_result = WaveArray::zeros(shape);
        domain.medium(&test_vec, &mut medium_result);

        let medium_norm = medium_result.norm_squared().sqrt() / test_vec.norm_squared().sqrt();
        println!("  ||B|| operator norm ≈ {:.4}", medium_norm);

        // Test if ||(I - B)|| < 1 for convergence
        test_vec -= medium_result;
        let i_minus_b_norm = test_vec.norm_squared().sqrt() / (shape.0 * shape.1 * shape.2) as f64;
        println!("  ||(I - B)|| ≈ {:.4}", i_minus_b_norm);

        println!();
    }

    println!("Analysis:");
    println!("- For homogeneous medium (0% perturbation), B = I exactly");
    println!("- For small perturbations, the scale factor approaches -i");
    println!("- This makes B_scat have large imaginary parts");
    println!("- The operator (I - B) becomes poorly conditioned");
    println!("- This explains the slow convergence for small perturbations");
}

#[test]
fn test_different_scaling_strategies() {
    println!("\n=== Testing Alternative Scaling Strategies ===\n");

    let shape = (4, 4, 4);
    let mut permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
    permittivity.data[[2, 2, 2]] = Complex::new(1.05, 0.0);

    // Test with different wavelengths (changes k0)
    let wavelengths = vec![0.5, 1.0, 2.0];

    for wavelength in wavelengths {
        let domain = HelmholtzDomain::new(
            permittivity.clone(),
            0.25,
            wavelength,
            [true, true, true],
            [[0, 0], [0, 0], [0, 0]],
        );

        println!("Wavelength = {:.1}:", wavelength);
        println!("  k0^2 = {:.2}", domain.k02.norm());
        println!("  Scale factor = {:?}", domain.scale());

        let mut source = WaveArray::zeros(shape);
        source.data[[2, 2, 2]] = Complex::new(1.0, 0.0);

        // Test with optimal alpha for this configuration
        let alphas = vec![0.1, 0.3, 0.5, 0.7];
        let mut best_residual = f64::INFINITY;
        let mut best_alpha = 0.0;

        for alpha in alphas {
            let config = IterationConfig {
                max_iterations: 200,
                threshold: 1e-4,
                alpha,
                full_residuals: false,
            };

            let result = preconditioned_richardson(&domain, &source, config);
            if result.residual_norm < best_residual {
                best_residual = result.residual_norm;
                best_alpha = alpha;
            }
        }

        println!(
            "  Best: alpha = {:.1}, residual = {:.2e}",
            best_alpha, best_residual
        );
        println!();
    }
}

#[test]
fn test_alternative_formulation() {
    println!("\n=== Testing Alternative Problem Formulation ===\n");

    // Instead of solving with small perturbation directly,
    // we could solve the scattered field problem

    let shape = (4, 4, 4);
    let mut permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
    permittivity.data[[2, 2, 2]] = Complex::new(1.05, 0.0);

    let domain = HelmholtzDomain::new(
        permittivity,
        0.25,
        1.0,
        [true, true, true],
        [[0, 0], [0, 0], [0, 0]],
    );

    // Test with incident field (plane wave approximation)
    let mut source = WaveArray::zeros(shape);
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                // Smooth source instead of point source
                let phase = 2.0 * std::f64::consts::PI * i as f64 / shape.0 as f64;
                source.data[[i, j, k]] = Complex::new(phase.cos() * 0.1, phase.sin() * 0.1);
            }
        }
    }

    let config = IterationConfig {
        max_iterations: 500,
        threshold: 1e-4,
        alpha: 0.4,
        full_residuals: false,
    };

    let result = preconditioned_richardson(&domain, &source, config);

    println!("Smooth source results:");
    println!("  Iterations: {}", result.iterations);
    println!("  Final residual: {:.2e}", result.residual_norm);
    println!();
    println!("Conclusion: Smooth sources may converge better than point sources");
}
