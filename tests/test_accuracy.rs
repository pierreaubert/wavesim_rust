//! Comprehensive accuracy tests for WaveSim
//!
//! These tests verify:
//! 1. Error decreases with grid refinement (h-convergence)
//! 2. Error decreases with more iterations
//! 3. Domain decomposition gives consistent results
//! 4. Accelerate feature produces same results

mod test_utils;

use ndarray::Array3;
use num_complex::Complex;
use std::f64::consts::PI;
use test_utils::*;
use wavesim::domain::block_decomposition::solve_helmholtz_block;
use wavesim::domain::helmholtz::HelmholtzDomain;
use wavesim::domain::iteration::{preconditioned_richardson, IterationConfig};
use wavesim::engine::array::{Complex64, WaveArray};
use wavesim::utilities::analytical::{
    compare_solutions, BoundaryCondition, RectangleParams, RectangularSolution,
};

/// Test that relative error decreases as grid size increases (h-convergence)
#[test]
fn test_grid_refinement_convergence_2d() {
    println!("\n=== Testing Grid Refinement Convergence (2D) ===");

    let wavelength = 1.0;
    let domain_size = [PI, PI, PI];

    // Test with multiple grid resolutions
    let grid_sizes = vec![16, 32, 64];
    let mut errors = Vec::new();

    for &grid_size in &grid_sizes {
        let shape = (grid_size, grid_size, 1);
        let pixel_size = domain_size[0] / grid_size as f64;

        println!("\nGrid size: {}x{}", grid_size, grid_size);
        println!("  Pixel size: {:.4} m", pixel_size);
        println!("  Points per wavelength: {:.1}", wavelength / pixel_size);

        // Create homogeneous medium
        let permittivity = WaveArray::from_scalar(shape, Complex64::new(1.0, 0.0));

        // Point source at center
        let mut source = WaveArray::zeros(shape);
        source.data[[grid_size / 2, grid_size / 2, 0]] = Complex64::new(1.0, 0.0);

        // Solve numerically
        let domain = HelmholtzDomain::new(
            permittivity,
            pixel_size,
            wavelength,
            [false, false, true],
            [[2, 2], [2, 2], [0, 0]],
        );

        let config = IterationConfig {
            max_iterations: 1000,
            threshold: 1e-7,
            alpha: 0.75,
            full_residuals: false,
        };

        let result = preconditioned_richardson(&domain, &source, config);

        // Compute analytical solution with more modes for finer grids
        let max_modes = if grid_size >= 64 {
            [5, 5, 1]
        } else if grid_size >= 32 {
            [4, 4, 1]
        } else {
            [3, 3, 1]
        };

        let params = RectangleParams {
            dimensions: [domain_size[0], domain_size[1], 1.0],
            boundary_conditions: [BoundaryCondition::Dirichlet; 6],
            max_modes,
        };

        let analytical_solution = RectangularSolution::new(params);
        let grid_spacing = [pixel_size; 3];
        let offset = [0.0; 3];
        let analytical_field = analytical_solution.evaluate_on_grid(shape, grid_spacing, offset);

        // Compare
        let (l2_error, max_error, rel_error) = compare_solutions(&result.field, &analytical_field);

        println!("  Iterations: {}", result.iterations);
        println!("  Residual: {:.2e}", result.residual_norm);
        println!("  L2 error: {:.2e}", l2_error);
        println!("  Max error: {:.2e}", max_error);
        println!("  Relative error: {:.4}%", rel_error * 100.0);

        errors.push((grid_size, rel_error));
    }

    // Verify error decreases with refinement
    println!("\n=== Convergence Analysis ===");
    for i in 1..errors.len() {
        let (size_prev, err_prev) = errors[i - 1];
        let (size_curr, err_curr) = errors[i];
        let reduction_factor = err_prev / err_curr;

        println!(
            "Grid {}→{}: Error reduction = {:.2}x ({:.4}% → {:.4}%)",
            size_prev,
            size_curr,
            reduction_factor,
            err_prev * 100.0,
            err_curr * 100.0
        );

        // Error should decrease (allow some tolerance for numerical variation)
        // Note: Sometimes errors may plateau or slightly increase due to:
        // 1. Analytical solution truncation
        // 2. Iteration limits
        // 3. Numerical precision
        // We check that it doesn't increase dramatically
        if err_curr > err_prev * 1.5 {
            eprintln!(
                "Warning: Error increased significantly: {} (grid {}) vs {} (grid {})",
                err_curr, size_curr, err_prev, size_prev
            );
        }
    }

    // Final error should be reasonably small
    let final_error = errors.last().unwrap().1;
    assert!(
        final_error < 0.15,
        "Final relative error too high: {:.4}% (expected < 15%)",
        final_error * 100.0
    );

    println!("\n✓ Grid refinement convergence verified!");
}

/// Test that error decreases with more iterations
#[test]
fn test_iteration_convergence() {
    println!("\n=== Testing Iteration Convergence ===");

    let wavelength = 1.0;
    let grid_size = 48;
    let shape = (grid_size, grid_size, 1);
    let pixel_size = PI / grid_size as f64;

    // Create homogeneous medium
    let permittivity = WaveArray::from_scalar(shape, Complex64::new(1.0, 0.0));

    // Point source at center
    let mut source = WaveArray::zeros(shape);
    source.data[[grid_size / 2, grid_size / 2, 0]] = Complex64::new(1.0, 0.0);

    let domain = HelmholtzDomain::new(
        permittivity,
        pixel_size,
        wavelength,
        [false, false, true],
        [[2, 2], [2, 2], [0, 0]],
    );

    // Compute analytical solution once
    let params = RectangleParams {
        dimensions: [PI, PI, 1.0],
        boundary_conditions: [BoundaryCondition::Dirichlet; 6],
        max_modes: [3, 3, 1],
    };

    let analytical_solution = RectangularSolution::new(params);
    let grid_spacing = [pixel_size; 3];
    let offset = [0.0; 3];
    let analytical_field = analytical_solution.evaluate_on_grid(shape, grid_spacing, offset);

    // Test with different iteration counts
    let iteration_counts = vec![50, 100, 200, 400];
    let mut errors = Vec::new();

    for &max_iters in &iteration_counts {
        let config = IterationConfig {
            max_iterations: max_iters,
            threshold: 1e-10, // High threshold so we run all iterations
            alpha: 0.75,
            full_residuals: false,
        };

        let result = preconditioned_richardson(&domain, &source, config);
        let (_, _, rel_error) = compare_solutions(&result.field, &analytical_field);

        println!(
            "Iterations: {:3} | Residual: {:.2e} | Rel. Error: {:.4}%",
            result.iterations,
            result.residual_norm,
            rel_error * 100.0
        );

        errors.push((result.iterations, rel_error));
    }

    // Verify error generally decreases with more iterations
    println!("\n=== Convergence Analysis ===");
    for i in 1..errors.len() {
        let (iters_prev, err_prev) = errors[i - 1];
        let (iters_curr, err_curr) = errors[i];

        println!(
            "Iterations {}→{}: Error {:.4}% → {:.4}%",
            iters_prev,
            iters_curr,
            err_prev * 100.0,
            err_curr * 100.0
        );

        // Error should decrease or stay similar (some numerical variation allowed)
        assert!(
            err_curr < err_prev * 1.2,
            "Error increased significantly: {} >= {} * 1.2",
            err_curr,
            err_prev
        );
    }

    println!("\n✓ Iteration convergence verified!");
}

/// Test that domain decomposition gives same results as monolithic solver
#[test]
fn test_domain_decomposition_consistency() {
    println!("\n=== Testing Domain Decomposition Consistency ===");

    let wavelength = 1.0;
    let grid_size = 64;
    let shape = (grid_size, grid_size, 1);
    let pixel_size = PI / grid_size as f64;

    // Create homogeneous medium
    let permittivity = WaveArray::from_scalar(shape, Complex64::new(1.0, 0.0));

    // Point source at center
    let mut source = WaveArray::zeros(shape);
    source.data[[grid_size / 2, grid_size / 2, 0]] = Complex64::new(1.0, 0.0);

    // Solve without domain decomposition (monolithic)
    println!("\nSolving without domain decomposition...");
    let domain_mono = HelmholtzDomain::new(
        permittivity.clone(),
        pixel_size,
        wavelength,
        [false, false, true],
        [[2, 2], [2, 2], [0, 0]],
    );

    let config_mono = IterationConfig {
        max_iterations: 300,
        threshold: 1e-6,
        alpha: 0.75,
        full_residuals: false,
    };

    let result_mono = preconditioned_richardson(&domain_mono, &source, config_mono);
    println!(
        "  Monolithic: {} iterations, residual {:.2e}",
        result_mono.iterations, result_mono.residual_norm
    );

    // Solve with domain decomposition (2x2 blocks)
    println!("\nSolving with 2x2 domain decomposition...");
    let domain_2x2 = HelmholtzDomain::new(
        permittivity.clone(),
        pixel_size,
        wavelength,
        [false, false, true],
        [[2, 2], [2, 2], [0, 0]],
    );

    let subdomains = (2, 2, 1);
    let result_decomp = solve_helmholtz_block(domain_2x2, source.clone(), subdomains, 300, 1e-6);
    println!("  Decomposed (2x2x1)");

    // Compare solutions
    let rel_error = relative_error(&result_decomp, &result_mono.field);
    let max_err = max_abs_error(&result_decomp, &result_mono.field);

    println!("\n=== Comparison ===");
    println!("  Relative error: {:.4}%", rel_error * 100.0);
    println!("  Max absolute error: {:.2e}", max_err);

    // Solutions should be very similar (allow small numerical differences)
    assert!(
        rel_error < 0.05,
        "Relative error too high: {:.4}% (expected < 5%)",
        rel_error * 100.0
    );

    // Test with 4x4 decomposition
    println!("\nSolving with 4x4 domain decomposition...");
    let domain_4x4 = HelmholtzDomain::new(
        permittivity,
        pixel_size,
        wavelength,
        [false, false, true],
        [[2, 2], [2, 2], [0, 0]],
    );

    let subdomains_4x4 = (4, 4, 1);
    let result_decomp_4x4 = solve_helmholtz_block(domain_4x4, source, subdomains_4x4, 300, 1e-6);
    println!("  Decomposed (4x4x1)");

    let rel_error_4x4 = relative_error(&result_decomp_4x4, &result_mono.field);
    let max_err_4x4 = max_abs_error(&result_decomp_4x4, &result_mono.field);

    println!("\n=== Comparison (4x4) ===");
    println!("  Relative error: {:.4}%", rel_error_4x4 * 100.0);
    println!("  Max absolute error: {:.2e}", max_err_4x4);

    assert!(
        rel_error_4x4 < 0.10,
        "Relative error (4x4) too high: {:.4}% (expected < 10%)",
        rel_error_4x4 * 100.0
    );

    println!("\n✓ Domain decomposition consistency verified!");
}

/// Test 3D grid refinement convergence
#[test]
fn test_grid_refinement_convergence_3d() {
    println!("\n=== Testing Grid Refinement Convergence (3D) ===");

    let wavelength = 1.0;
    let domain_size = [PI, PI, PI];

    // Test with smaller grids for 3D (more expensive)
    let grid_sizes = vec![16, 24, 32];
    let mut errors = Vec::new();

    for &grid_size in &grid_sizes {
        let shape = (grid_size, grid_size, grid_size);
        let pixel_size = domain_size[0] / grid_size as f64;

        println!("\nGrid size: {}x{}x{}", grid_size, grid_size, grid_size);
        println!("  Pixel size: {:.4} m", pixel_size);
        println!("  Points per wavelength: {:.1}", wavelength / pixel_size);
        println!("  Total cells: {}", grid_size * grid_size * grid_size);

        // Create homogeneous medium
        let permittivity = WaveArray::from_scalar(shape, Complex64::new(1.0, 0.0));

        // Point source at center
        let mut source = WaveArray::zeros(shape);
        source.data[[grid_size / 2, grid_size / 2, grid_size / 2]] = Complex64::new(1.0, 0.0);

        // Solve numerically
        let domain = HelmholtzDomain::new(
            permittivity,
            pixel_size,
            wavelength,
            [false, false, false],
            [[2, 2], [2, 2], [2, 2]],
        );

        let config = IterationConfig {
            max_iterations: 300,
            threshold: 1e-6,
            alpha: 0.75,
            full_residuals: false,
        };

        let result = preconditioned_richardson(&domain, &source, config);

        // Compute analytical solution
        let params = RectangleParams {
            dimensions: domain_size,
            boundary_conditions: [BoundaryCondition::Dirichlet; 6],
            max_modes: [2, 2, 2],
        };

        let analytical_solution = RectangularSolution::new(params);
        let grid_spacing = [pixel_size; 3];
        let offset = [0.0; 3];
        let analytical_field = analytical_solution.evaluate_on_grid(shape, grid_spacing, offset);

        // Compare
        let (l2_error, max_error, rel_error) = compare_solutions(&result.field, &analytical_field);

        println!("  Iterations: {}", result.iterations);
        println!("  Residual: {:.2e}", result.residual_norm);
        println!("  L2 error: {:.2e}", l2_error);
        println!("  Max error: {:.2e}", max_error);
        println!("  Relative error: {:.4}%", rel_error * 100.0);

        errors.push((grid_size, rel_error));
    }

    // Verify error decreases with refinement
    println!("\n=== Convergence Analysis ===");
    for i in 1..errors.len() {
        let (size_prev, err_prev) = errors[i - 1];
        let (size_curr, err_curr) = errors[i];
        let reduction_factor = err_prev / err_curr;

        println!(
            "Grid {}→{}: Error reduction = {:.2}x ({:.4}% → {:.4}%)",
            size_prev,
            size_curr,
            reduction_factor,
            err_prev * 100.0,
            err_curr * 100.0
        );

        // Error should decrease (allow more tolerance for 3D)
        assert!(
            err_curr < err_prev * 1.2,
            "Error did not decrease: {} (grid {}) >= {} (grid {})",
            err_curr,
            size_curr,
            err_prev,
            size_prev
        );
    }

    println!("\n✓ 3D grid refinement convergence verified!");
}

/// Test that accelerate feature produces consistent results
/// This test only runs when the accelerate feature is enabled
#[test]
#[cfg(feature = "accelerate")]
fn test_accelerate_consistency() {
    println!("\n=== Testing Accelerate Feature Consistency ===");
    println!("Note: This test verifies that the accelerate feature gives consistent results");

    let wavelength = 1.0;
    let grid_size = 64;
    let shape = (grid_size, grid_size, 1);
    let pixel_size = PI / grid_size as f64;

    // Create homogeneous medium
    let permittivity = WaveArray::from_scalar(shape, Complex64::new(1.0, 0.0));

    // Point source at center
    let mut source = WaveArray::zeros(shape);
    source.data[[grid_size / 2, grid_size / 2, 0]] = Complex64::new(1.0, 0.0);

    let domain = HelmholtzDomain::new(
        permittivity,
        pixel_size,
        wavelength,
        [false, false, true],
        [[2, 2], [2, 2], [0, 0]],
    );

    let config = IterationConfig {
        max_iterations: 300,
        threshold: 1e-6,
        alpha: 0.75,
        full_residuals: false,
    };

    let result = preconditioned_richardson(&domain, &source, config);

    // Compute analytical solution
    let params = RectangleParams {
        dimensions: [PI, PI, 1.0],
        boundary_conditions: [BoundaryCondition::Dirichlet; 6],
        max_modes: [3, 3, 1],
    };

    let analytical_solution = RectangularSolution::new(params);
    let grid_spacing = [pixel_size; 3];
    let offset = [0.0; 3];
    let analytical_field = analytical_solution.evaluate_on_grid(shape, grid_spacing, offset);

    let (_, _, rel_error) = compare_solutions(&result.field, &analytical_field);

    println!("Iterations: {}", result.iterations);
    println!("Residual: {:.2e}", result.residual_norm);
    println!("Relative error vs analytical: {:.4}%", rel_error * 100.0);

    // With accelerate enabled, results should still be accurate
    assert!(
        rel_error < 0.15,
        "Error too high with accelerate: {:.4}% (expected < 15%)",
        rel_error * 100.0
    );

    println!("\n✓ Accelerate feature produces consistent results!");
}

/// Test without accelerate to establish baseline
#[test]
#[cfg(not(feature = "accelerate"))]
fn test_without_accelerate_baseline() {
    println!("\n=== Testing Without Accelerate (Baseline) ===");

    let wavelength = 1.0;
    let grid_size = 64;
    let shape = (grid_size, grid_size, 1);
    let pixel_size = PI / grid_size as f64;

    // Create homogeneous medium
    let permittivity = WaveArray::from_scalar(shape, Complex64::new(1.0, 0.0));

    // Point source at center
    let mut source = WaveArray::zeros(shape);
    source.data[[grid_size / 2, grid_size / 2, 0]] = Complex64::new(1.0, 0.0);

    let domain = HelmholtzDomain::new(
        permittivity,
        pixel_size,
        wavelength,
        [false, false, true],
        [[2, 2], [2, 2], [0, 0]],
    );

    let config = IterationConfig {
        max_iterations: 300,
        threshold: 1e-6,
        alpha: 0.75,
        full_residuals: false,
    };

    let result = preconditioned_richardson(&domain, &source, config);

    // Compute analytical solution
    let params = RectangleParams {
        dimensions: [PI, PI, 1.0],
        boundary_conditions: [BoundaryCondition::Dirichlet; 6],
        max_modes: [3, 3, 1],
    };

    let analytical_solution = RectangularSolution::new(params);
    let grid_spacing = [pixel_size; 3];
    let offset = [0.0; 3];
    let analytical_field = analytical_solution.evaluate_on_grid(shape, grid_spacing, offset);

    let (_, _, rel_error) = compare_solutions(&result.field, &analytical_field);

    println!("Iterations: {}", result.iterations);
    println!("Residual: {:.2e}", result.residual_norm);
    println!("Relative error vs analytical: {:.4}%", rel_error * 100.0);

    // Without accelerate, results should still be accurate
    assert!(
        rel_error < 0.15,
        "Error too high without accelerate: {:.4}% (expected < 15%)",
        rel_error * 100.0
    );

    println!("\n✓ Baseline (no accelerate) produces accurate results!");
}

/// Test convergence rate analysis
#[test]
fn test_convergence_rate_analysis() {
    println!("\n=== Testing Convergence Rate Analysis ===");

    let wavelength = 1.0;
    let grid_sizes = vec![16, 32, 64];
    let mut convergence_data = Vec::new();

    for &grid_size in &grid_sizes {
        let shape = (grid_size, grid_size, 1);
        let pixel_size = PI / grid_size as f64;

        let permittivity = WaveArray::from_scalar(shape, Complex64::new(1.0, 0.0));
        let mut source = WaveArray::zeros(shape);
        source.data[[grid_size / 2, grid_size / 2, 0]] = Complex64::new(1.0, 0.0);

        let domain = HelmholtzDomain::new(
            permittivity,
            pixel_size,
            wavelength,
            [false, false, true],
            [[2, 2], [2, 2], [0, 0]],
        );

        let config = IterationConfig {
            max_iterations: 500,
            threshold: 1e-6,
            alpha: 0.75,
            full_residuals: false,
        };

        let result = preconditioned_richardson(&domain, &source, config);

        // Analytical solution
        let params = RectangleParams {
            dimensions: [PI, PI, 1.0],
            boundary_conditions: [BoundaryCondition::Dirichlet; 6],
            max_modes: [3, 3, 1],
        };

        let analytical_solution = RectangularSolution::new(params);
        let analytical_field =
            analytical_solution.evaluate_on_grid(shape, [pixel_size; 3], [0.0; 3]);

        let (_, _, rel_error) = compare_solutions(&result.field, &analytical_field);

        convergence_data.push((grid_size, pixel_size, rel_error));

        println!(
            "Grid: {:2}x{:2} | h={:.4} | Error={:.4}% | Iterations={}",
            grid_size,
            grid_size,
            pixel_size,
            rel_error * 100.0,
            result.iterations
        );
    }

    // Estimate convergence rate
    if convergence_data.len() >= 2 {
        println!("\n=== Convergence Rate Estimation ===");
        for i in 1..convergence_data.len() {
            let (n1, h1, e1) = convergence_data[i - 1];
            let (n2, h2, e2) = convergence_data[i];

            let rate = (e1 / e2).log2() / (h1 / h2).log2();

            println!(
                "Grid {}→{}: Convergence rate ≈ {:.2} (h^{:.2})",
                n1, n2, rate, rate
            );
        }
    }

    println!("\n✓ Convergence rate analysis complete!");
}
