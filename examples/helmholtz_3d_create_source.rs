//! Helmholtz 3D example demonstrating different source types
//!
//! This example simulates the propagation of a wave through a 3D medium
//! with an interface between two media with different refractive indices,
//! demonstrating different source types: point source and Gaussian beam.

use ndarray::Array3;
use num_complex::Complex;
use std::time::Instant;
use wavesim::prelude::*;

/// Create a Gaussian source distribution
fn create_gaussian_source(
    shape: (usize, usize, usize),
    center: [f64; 3],
    width: [f64; 3],
    amplitude: Complex<f64>,
) -> Array3<Complex<f64>> {
    let mut source = Array3::zeros(shape);

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                let dx = i as f64 - center[0];
                let dy = j as f64 - center[1];
                let dz = k as f64 - center[2];

                let r2 =
                    (dx / width[0]).powi(2) + (dy / width[1]).powi(2) + (dz / width[2]).powi(2);

                source[[i, j, k]] = amplitude * (-r2).exp();
            }
        }
    }

    source
}

fn main() {
    println!("Helmholtz 3D Source Types Demonstration");
    println!("========================================\n");

    // Parameters
    let wavelength = 1.0; // Wavelength in micrometers (μm)
    let pixel_size = wavelength / 8.0; // Pixel size in micrometers

    // Create domain size
    let sim_size = [10.0, 8.0, 5.0]; // Physical size in micrometers
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
        "  Physical size: {:.1}x{:.1}x{:.1} μm³\n",
        sim_size[0], sim_size[1], sim_size[2]
    );

    // Create refractive index map with two regions
    let mut refractive_index = Array3::from_elem(
        (n_size[0], n_size[1], n_size[2]),
        Complex::new(1.0, 0.0), // Background n=1
    );

    // Create interface: lower half has n=2
    for i in 0..n_size[0] {
        for j in n_size[1] / 2..n_size[1] {
            for k in 0..n_size[2] {
                refractive_index[[i, j, k]] = Complex::new(2.0, 0.0);
            }
        }
    }

    // Convert refractive index to permittivity (n² = ε)
    let permittivity = refractive_index.map(|n| n * n);

    println!("Medium configuration:");
    println!("  Upper half: n = 1.0 (air/vacuum)");
    println!("  Lower half: n = 2.0 (dielectric)\n");

    // Choose source type
    let source_type = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "point".to_string());

    let source = match source_type.as_str() {
        "gaussian" | "gauss" => {
            println!("Using GAUSSIAN source");

            // Create Gaussian beam source
            let source_shape = (n_size[0] / 2, 1, n_size[2]);
            let source_data = create_gaussian_source(
                source_shape,
                [
                    source_shape.0 as f64 / 2.0,
                    0.0,
                    source_shape.2 as f64 / 2.0,
                ],
                [
                    source_shape.0 as f64 / 6.0,
                    1.0,
                    source_shape.2 as f64 / 6.0,
                ],
                Complex::new(2.0, 0.0),
            );

            println!("  Shape: {:?} pixels", source_shape);
            println!(
                "  Width: ~{:.1} μm (FWHM)",
                source_shape.0 as f64 * pixel_size / 3.0
            );
            println!("  Position: left edge at interface");
            println!("  Amplitude: 2.0\n");

            Source::from_data(
                WaveArray { data: source_data },
                [0, n_size[1] / 2 - 1, 0], // Position at interface
            )
        }
        _ => {
            println!("Using POINT source (default)");
            println!("  Position: left edge at interface");
            println!("  Amplitude: 3.0 + 0.5i\n");

            Source::point([0, n_size[1] / 2, n_size[2] / 2], Complex::new(3.0, 0.5))
        }
    };

    println!("Tip: Run with 'gaussian' argument for Gaussian source");
    println!("     cargo run --release --example helmholtz_3d_create_source gaussian\n");

    // Configure simulation
    let params = SimulationParams {
        wavelength,
        pixel_size,
        boundary_width: 10.0 * pixel_size, // 10 pixels of absorbing boundary
        periodic: [false, false, false],
        max_iterations: 200, // Reduced for faster demo
        threshold: 1e-4,
        alpha: 0.75,
        full_residuals: false,
        crop_boundaries: true,
        n_domains: None,
    };

    println!("Running simulation...");
    let start = Instant::now();

    // Run the simulation
    let result = simulate(permittivity, vec![source], params);

    let duration = start.elapsed();
    println!("\nSimulation completed in {:.2} s", duration.as_secs_f64());
    println!("  Iterations: {}", result.iterations);
    println!(
        "  Time per iteration: {:.4} s",
        duration.as_secs_f64() / result.iterations as f64
    );
    println!("  Residual norm: {:.2e}\n", result.residual_norm);

    // Analyze the field
    let field_shape = result.field.shape();
    println!("Field analysis:");
    println!("  Field shape: {:?}", field_shape);

    // Calculate field statistics
    let field_norm = result.field.norm_squared().sqrt();
    let mut max_amplitude = 0.0;
    let mut max_position = [0, 0, 0];

    for i in 0..field_shape[0] {
        for j in 0..field_shape[1] {
            for k in 0..field_shape[2] {
                let amplitude = result.field.data[[i, j, k]].norm();
                if amplitude > max_amplitude {
                    max_amplitude = amplitude;
                    max_position = [i, j, k];
                }
            }
        }
    }

    println!("  Field L2 norm: {:.3}", field_norm);
    println!("  Max amplitude: {:.3e}", max_amplitude);
    println!("  Max position: {:?} (pixels)", max_position);

    // Check field at interface
    let interface_y = field_shape[1] / 2;
    let interface_point = [field_shape[0] / 2, interface_y, field_shape[2] / 2];
    let interface_value =
        result.field.data[[interface_point[0], interface_point[1], interface_point[2]]];

    println!("\n  Interface field (center):");
    println!("    Position: {:?}", interface_point);
    println!(
        "    Value: {:.3e} + {:.3e}i",
        interface_value.re, interface_value.im
    );
    println!("    Amplitude: {:.3e}", interface_value.norm());

    // Check transmission/reflection characteristics
    let mut upper_energy = 0.0;
    let mut lower_energy = 0.0;

    for i in 0..field_shape[0] {
        for j in 0..field_shape[1] {
            for k in 0..field_shape[2] {
                let energy = result.field.data[[i, j, k]].norm_sqr();
                if j < interface_y {
                    upper_energy += energy;
                } else {
                    lower_energy += energy;
                }
            }
        }
    }

    let total_energy = upper_energy + lower_energy;
    if total_energy > 0.0 {
        println!("\n  Energy distribution:");
        println!(
            "    Upper medium (n=1): {:.1}%",
            100.0 * upper_energy / total_energy
        );
        println!(
            "    Lower medium (n=2): {:.1}%",
            100.0 * lower_energy / total_energy
        );
    }

    println!("\n✓ Simulation completed successfully!");
}
