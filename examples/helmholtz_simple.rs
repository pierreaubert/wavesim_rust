//! Simple example demonstrating the Helmholtz solver
//!
//! This example shows how to:
//! - Create a simple medium (homogeneous)
//! - Add a point source
//! - Run the simulation
//! - Display basic results

use ndarray::Array3;
use num_complex::Complex;
use wavesim::prelude::*;

fn main() {
    println!("WaveSim Rust - Simple Helmholtz Example");
    println!("========================================\n");

    // Simulation parameters
    let wavelength = 0.5; // micrometers
    let pixel_size = wavelength / 4.0; // λ/4 sampling
    let domain_size = 64; // 64x64x64 pixels

    println!("Configuration:");
    println!("  Wavelength: {} μm", wavelength);
    println!("  Pixel size: {} μm", pixel_size);
    println!(
        "  Domain size: {}x{}x{} pixels",
        domain_size, domain_size, domain_size
    );
    println!(
        "  Physical size: {:.1}x{:.1}x{:.1} μm³\n",
        domain_size as f64 * pixel_size,
        domain_size as f64 * pixel_size,
        domain_size as f64 * pixel_size
    );

    // Create a homogeneous medium (n=1.0 everywhere)
    let permittivity = Array3::from_elem(
        (domain_size, domain_size, domain_size),
        Complex::new(1.0, 0.0),
    );

    // Create a point source at the center
    let source = Source::point(
        [domain_size / 2, domain_size / 2, domain_size / 2],
        Complex::new(1.0, 0.0),
    );

    // Configure simulation
    let params = SimulationParams {
        wavelength,
        pixel_size,
        boundary_width: wavelength * 2.0, // 2λ absorbing boundaries
        periodic: [false, false, false],
        max_iterations: 100,
        threshold: 1e-4,
        alpha: 0.75,
        full_residuals: false,
        crop_boundaries: true,
        n_domains: None, // Single domain
    };

    println!("Running simulation...");
    let start = std::time::Instant::now();

    // Run the simulation
    let result = simulate(permittivity, vec![source], params);

    let duration = start.elapsed();
    println!("Simulation completed in {:.2?}", duration);
    println!("\nResults:");
    println!("  Iterations: {}", result.iterations);
    println!("  Residual norm: {:.2e}", result.residual_norm);
    println!("  Field shape: {:?}", result.field.shape());

    // Calculate field statistics
    let field_norm = result.field.norm_squared().sqrt();
    let max_amplitude = result
        .field
        .data
        .iter()
        .map(|c| c.norm())
        .fold(0.0, f64::max);

    println!("  Field norm: {:.3}", field_norm);
    println!("  Max amplitude: {:.3}", max_amplitude);

    // Check center value (should be highest for point source in homogeneous medium)
    let center = [
        result.field.shape()[0] / 2,
        result.field.shape()[1] / 2,
        result.field.shape()[2] / 2,
    ];
    let center_value = result.field.data[[center[0], center[1], center[2]]];
    println!(
        "  Center field value: {:.3} + {:.3}i",
        center_value.re, center_value.im
    );
    println!("  Center amplitude: {:.3}", center_value.norm());

    println!("\n✓ Simulation successful!");
}
