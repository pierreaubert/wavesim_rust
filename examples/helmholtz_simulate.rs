//! Helmholtz simulation with full parameter configuration
//!
//! This example demonstrates all the arguments of the simulate wrapper function
//! through an example solving the Helmholtz equation with random permittivity.

use ndarray::Array3;
use num_complex::Complex;
use rand::prelude::*;
use std::time::Instant;
use wavesim::prelude::*;

/// Generate random complex permittivity for testing
fn random_permittivity(shape: (usize, usize, usize)) -> Array3<Complex<f64>> {
    let mut rng = thread_rng();
    let mut perm = Array3::zeros(shape);

    for elem in perm.iter_mut() {
        // Random refractive index between 1.0 and 2.0
        let n_real = 1.0 + rng.gen::<f64>();
        // Small imaginary part for absorption
        let n_imag = rng.gen::<f64>() * 0.1;
        // permittivity = n²
        let n = Complex::new(n_real, n_imag);
        *elem = n * n;
    }

    perm
}

fn main() {
    println!("Helmholtz Simulate - Full Parameter Demonstration");
    println!("==================================================\n");

    // Parameters
    let wavelength = 1.0; // Wavelength in micrometers (μm)
    let pixel_size = wavelength / 4.0; // Pixel size in micrometers

    // Size of simulation domain in micrometers
    let sim_size = [8.0, 5.0, 10.0];
    let n_size = [
        (sim_size[0] / pixel_size) as usize,
        (sim_size[1] / pixel_size) as usize,
        (sim_size[2] / pixel_size) as usize,
    ];

    println!("Configuration:");
    println!("  Wavelength: {} μm", wavelength);
    println!("  Pixel size: {} μm", pixel_size);
    println!("  Domain size: {:?} pixels", n_size);
    println!(
        "  Physical size: {:.1}x{:.1}x{:.1} μm³",
        sim_size[0], sim_size[1], sim_size[2]
    );

    // Create random permittivity map
    let permittivity = random_permittivity((n_size[0], n_size[1], n_size[2]));

    // Create a point source at the center
    let source = Source::point(
        [n_size[0] / 2, n_size[1] / 2, n_size[2] / 2],
        Complex::new(1.0, 0.0),
    );

    println!("\nSimulation parameters:");
    println!("  Boundary width: 5.0 μm");
    println!("  Periodic: (false, false, false)");
    println!("  Max iterations: 1000");
    println!("  Threshold: 1e-7");
    println!("  Alpha: 0.9");
    println!("  Full residuals: true");
    println!("  Crop boundaries: false");
    println!("  Domains: [1, 1, 3] (domain decomposition)\n");

    // Configure simulation with all parameters
    let params = SimulationParams {
        wavelength,
        pixel_size,
        boundary_width: 5.0, // Boundary width in micrometers
        periodic: [false, false, false],
        max_iterations: 1000,
        threshold: 1e-7,            // Tighter convergence tolerance
        alpha: 0.9,                 // Higher relaxation factor
        full_residuals: true,       // Return full residual history
        crop_boundaries: false,     // Keep absorbing boundaries in result
        n_domains: Some([1, 1, 3]), // Domain decomposition
    };

    println!("Running simulation...");
    let start = Instant::now();

    // Run the simulation
    let result = simulate(permittivity, vec![source], params);

    let duration = start.elapsed();
    let time_per_iter = duration.as_secs_f64() / result.iterations as f64;

    println!("\n=== Results ===");
    println!("  Time: {:.2} s", duration.as_secs_f64());
    println!("  Iterations: {}", result.iterations);
    println!("  Time per iteration: {:.4} s", time_per_iter);
    println!("  Final residual norm: {:.2e}", result.residual_norm);
    println!("  Field shape: {:?}", result.field.shape());

    // Show residual history if available
    if let Some(ref history) = result.residual_history {
        println!("\nResidual convergence:");
        let n = history.len();
        let indices = [0, n / 4, n / 2, 3 * n / 4, n - 1];
        for &i in &indices {
            if i < n {
                println!("  Iteration {:4}: {:.2e}", i + 1, history[i]);
            }
        }

        // Check convergence rate
        if n > 10 {
            let early_avg = history[..5].iter().sum::<f64>() / 5.0;
            let late_avg = history[n - 5..].iter().sum::<f64>() / 5.0;
            let reduction_factor = early_avg / late_avg;
            println!("\n  Residual reduction factor: {:.1}x", reduction_factor);
        }
    }

    // Analyze field statistics
    let field_norm = result.field.norm_squared().sqrt();
    let max_amplitude = result
        .field
        .data
        .iter()
        .map(|c| c.norm())
        .fold(0.0, f64::max);
    let min_amplitude = result
        .field
        .data
        .iter()
        .map(|c| c.norm())
        .fold(f64::INFINITY, f64::min);

    println!("\nField statistics:");
    println!("  Field L2 norm: {:.3}", field_norm);
    println!("  Max amplitude: {:.3e}", max_amplitude);
    println!("  Min amplitude: {:.3e}", min_amplitude);
    println!(
        "  Dynamic range: {:.1} dB",
        20.0 * (max_amplitude / min_amplitude).log10()
    );

    println!("\n✓ Simulation completed successfully!");
}
