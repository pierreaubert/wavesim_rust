//! Integration tests for the WaveSim library

use ndarray::Array3;
use num_complex::Complex;
use wavesim::prelude::*;

#[test]
fn test_homogeneous_medium_convergence() {
    // Test that a homogeneous medium converges properly
    let size = 32;
    let permittivity = Array3::from_elem((size, size, size), Complex::new(1.0, 0.0));

    let source = Source::point([size / 2, size / 2, size / 2], Complex::new(1.0, 0.0));

    let params = SimulationParams {
        wavelength: 0.5,
        pixel_size: 0.125,
        boundary_width: 0.5,
        periodic: [false, false, false],
        max_iterations: 200,
        threshold: 1e-5,
        alpha: 0.75,
        full_residuals: false,
        crop_boundaries: true,
        n_domains: None,
    };

    let result = simulate(permittivity, vec![source], params);

    // Should converge
    assert!(
        result.iterations < 200,
        "Did not converge in 200 iterations"
    );
    assert!(
        result.residual_norm < 1e-4,
        "Residual too high: {}",
        result.residual_norm
    );

    // Field should be non-zero
    assert!(result.field.norm_squared() > 0.0);

    // Center should have maximum amplitude (point source in homogeneous medium)
    let center_idx = [size / 2, size / 2, size / 2];
    let center_amp = result.field.data[[center_idx[0], center_idx[1], center_idx[2]]].norm();

    // Check a few other points - they should have lower amplitude
    let corner_amp = result.field.data[[0, 0, 0]].norm();
    assert!(
        center_amp > corner_amp,
        "Center amplitude should be highest"
    );
}

#[test]
fn test_inhomogeneous_medium() {
    // Test with a lens-like structure
    let size = 32;
    let mut permittivity = Array3::from_elem((size, size, size), Complex::new(1.0, 0.0));

    // Create a spherical region with higher refractive index
    let center = size as f64 / 2.0;
    let radius = 5.0;
    for i in 0..size {
        for j in 0..size {
            for k in 0..size {
                let r = ((i as f64 - center).powi(2)
                    + (j as f64 - center).powi(2)
                    + (k as f64 - center).powi(2))
                .sqrt();
                if r < radius {
                    // n = 1.5, so n² = 2.25
                    permittivity[[i, j, k]] = Complex::new(2.25, 0.0);
                }
            }
        }
    }

    let source = Source::point(
        [5, size / 2, size / 2], // Source on one side
        Complex::new(1.0, 0.0),
    );

    let params = SimulationParams {
        wavelength: 0.5,
        pixel_size: 0.125,
        boundary_width: 0.5,
        periodic: [false, false, false],
        max_iterations: 500,
        threshold: 1e-4,
        alpha: 0.75,
        full_residuals: false,
        crop_boundaries: true,
        n_domains: None,
    };

    let result = simulate(permittivity, vec![source], params);

    // Should converge (may take more iterations due to inhomogeneity)
    assert!(
        result.iterations < 500,
        "Did not converge in 500 iterations"
    );
    assert!(result.field.norm_squared() > 0.0);
}

#[test]
#[ignore = "Periodic boundaries need more work"]
fn test_periodic_boundaries_symmetry() {
    // Test that periodic boundaries maintain symmetry
    let size = 16;
    let permittivity = Array3::from_elem((size, size, size), Complex::new(1.0, 0.0));

    // Source at the center
    let source = Source::point([size / 2, size / 2, size / 2], Complex::new(1.0, 0.0));

    let params = SimulationParams {
        wavelength: 0.5,
        pixel_size: 0.125,
        boundary_width: 0.0,            // No absorbing boundaries for periodic
        periodic: [true, false, false], // Periodic only in x
        max_iterations: 200,
        threshold: 1e-4,
        alpha: 0.75,
        full_residuals: false,
        crop_boundaries: false,
        n_domains: None,
    };

    let result = simulate(permittivity, vec![source], params);

    assert!(result.iterations < 200);
    assert!(result.field.norm_squared() > 0.0);

    // Check periodicity: field at x=0 should match x=size-1
    let tolerance = 1e-6;
    for j in 0..size {
        for k in 0..size {
            let diff = (result.field.data[[0, j, k]] - result.field.data[[size - 1, j, k]]).norm();
            assert!(
                diff < tolerance,
                "Periodic boundary violation at [{},{}]: diff = {}",
                j,
                k,
                diff
            );
        }
    }
}

#[test]
fn test_multiple_sources_superposition() {
    // Test that multiple sources combine correctly
    let size = 32;
    let permittivity = Array3::from_elem((size, size, size), Complex::new(1.0, 0.0));

    // Two sources with different amplitudes
    let sources = vec![
        Source::point([10, size / 2, size / 2], Complex::new(1.0, 0.0)),
        Source::point([22, size / 2, size / 2], Complex::new(0.5, 0.0)),
    ];

    let params = SimulationParams {
        wavelength: 0.5,
        pixel_size: 0.125,
        boundary_width: 0.5,
        periodic: [false, false, false],
        max_iterations: 200,
        threshold: 1e-4,
        alpha: 0.75,
        full_residuals: false,
        crop_boundaries: true,
        n_domains: None,
    };

    let result = simulate(permittivity, sources, params);

    assert!(result.iterations < 200);
    assert!(result.field.norm_squared() > 0.0);

    // Field should have contributions from both sources
    let field1 = result.field.data[[10, size / 2, size / 2]].norm();
    let field2 = result.field.data[[22, size / 2, size / 2]].norm();

    assert!(field1 > 0.0, "No field at first source position");
    assert!(field2 > 0.0, "No field at second source position");
}

#[test]
fn test_convergence_monitoring() {
    // Test that residual history is properly recorded
    let size = 16;
    let permittivity = Array3::from_elem((size, size, size), Complex::new(1.0, 0.0));

    let source = Source::point([size / 2, size / 2, size / 2], Complex::new(1.0, 0.0));

    let params = SimulationParams {
        wavelength: 0.5,
        pixel_size: 0.125,
        boundary_width: 0.5,
        periodic: [false, false, false],
        max_iterations: 50,
        threshold: 1e-4,
        alpha: 0.75,
        full_residuals: true, // Request full history
        crop_boundaries: true,
        n_domains: None,
    };

    let result = simulate(permittivity, vec![source], params);

    // Check that we got residual history
    assert!(result.residual_history.is_some());

    if let Some(history) = result.residual_history {
        assert_eq!(history.len(), result.iterations);

        // Residuals should generally decrease
        let window_size = 5;
        if history.len() > window_size * 2 {
            let early_avg: f64 = history[0..window_size].iter().sum::<f64>() / window_size as f64;
            let late_avg: f64 =
                history[history.len() - window_size..].iter().sum::<f64>() / window_size as f64;
            assert!(
                late_avg < early_avg,
                "Residuals not decreasing: early={:.2e}, late={:.2e}",
                early_avg,
                late_avg
            );
        }
    }
}

#[test]
fn test_physical_consistency() {
    // Test that the solution has expected physical properties
    let size = 32;
    let permittivity = Array3::from_elem((size, size, size), Complex::new(1.0, 0.0));

    let source = Source::point([size / 2, size / 2, size / 2], Complex::new(1.0, 0.0));

    let params = SimulationParams {
        wavelength: 0.5,
        pixel_size: 0.125, // λ/4
        boundary_width: 1.0,
        periodic: [false, false, false],
        max_iterations: 200,
        threshold: 1e-5,
        alpha: 0.75,
        full_residuals: false,
        crop_boundaries: true,
        n_domains: None,
    };

    let result = simulate(permittivity, vec![source], params);

    assert!(result.iterations < 200);

    // The field should decay with distance from source
    let center = [size / 2, size / 2, size / 2];
    let center_amp = result.field.data[[center[0], center[1], center[2]]].norm();

    // Check amplitudes at different distances
    let r1_amp = result.field.data[[center[0] + 2, center[1], center[2]]].norm();
    let r2_amp = result.field.data[[center[0] + 4, center[1], center[2]]].norm();

    // Amplitude should decrease with distance (in homogeneous medium)
    assert!(
        center_amp > r1_amp,
        "Amplitude should decrease with distance"
    );
    assert!(r1_amp > r2_amp, "Amplitude should decrease monotonically");
}
