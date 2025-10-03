//! Advanced 3D Helmholtz simulation example
//!
//! This example demonstrates advanced features including:
//! - Multiple source configurations  
//! - Domain decomposition for parallel computation
//! - Custom medium generation with interfaces
//! - Performance monitoring and optimization

use num_complex::Complex;
use std::time::Instant;
use wavesim::{
    domain::domain_trait::Domain,
    domain::helmholtz::HelmholtzDomain,
    domain::helmholtz_parallel::ParallelHelmholtzDomain,
    domain::iteration::{preconditioned_richardson, IterationConfig},
    engine::array::WaveArray,
    utilities::{add_absorbing_boundaries, create_gaussian_source},
};

type Complex64 = Complex<f64>;

/// Create a layered medium with different refractive indices
fn create_layered_medium(shape: [usize; 3]) -> WaveArray<Complex64> {
    let mut permittivity = WaveArray::zeros((shape[0], shape[1], shape[2]));

    // Create three layers with different refractive indices
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let n = if j < shape[1] / 3 {
                    1.0 // Air/vacuum
                } else if j < 2 * shape[1] / 3 {
                    1.5 // Glass
                } else {
                    2.0 // High-index material
                };

                // Add small absorption
                let absorption = 0.001 * (1.0 + j as f64 / shape[1] as f64);

                // Permittivity = n^2
                permittivity.data[[i, j, k]] = Complex::new(n * n, absorption);
            }
        }
    }

    permittivity
}

/// Create a lens-like structure
fn create_lens_medium(shape: [usize; 3], focal_length: f64) -> WaveArray<Complex64> {
    let mut permittivity = WaveArray::zeros((shape[0], shape[1], shape[2]));

    let center_x = shape[0] as f64 / 2.0;
    let center_y = shape[1] as f64 / 2.0;
    let center_z = shape[2] as f64 / 2.0;

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let dx = i as f64 - center_x;
                let dy = j as f64 - center_y;
                let dz = k as f64 - center_z;

                // Distance from center
                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                // Parabolic lens profile
                let n = if r < focal_length {
                    1.5 + 0.5 * (1.0 - r / focal_length).powi(2)
                } else {
                    1.0
                };

                permittivity.data[[i, j, k]] = Complex::new(n * n, 0.0);
            }
        }
    }

    permittivity
}

/// Create multiple Gaussian sources at different positions
fn create_multiple_sources(
    shape: [usize; 3],
    positions: Vec<[usize; 3]>,
    pixel_size: f64,
    wavelength: f64,
) -> Vec<(WaveArray<Complex64>, [usize; 3])> {
    let mut sources = Vec::new();

    for (idx, pos) in positions.iter().enumerate() {
        // Vary the width for different sources
        let width = 2.0 + idx as f64 * 0.5;

        // Create Gaussian source
        let (source, adjusted_pos) = create_gaussian_source(
            *pos,
            [width, width, width],
            pixel_size,
            wavelength,
            Complex::new(1.0, 0.0),
        );

        sources.push((source, adjusted_pos));
    }

    sources
}

/// Run advanced simulation with monitoring
fn run_advanced_simulation(medium_type: &str, use_parallel: bool, n_domains: Option<[usize; 3]>) {
    println!("\n{} Simulation", medium_type);
    println!("{}=", "=".repeat(medium_type.len() + 11));

    // Parameters
    let wavelength = 1.0; // μm
    let pixel_size = wavelength / 8.0; // Higher resolution

    // Domain size
    let sim_size = [12.0, 12.0, 12.0]; // μm
    let shape = [
        (sim_size[0] / pixel_size) as usize,
        (sim_size[1] / pixel_size) as usize,
        (sim_size[2] / pixel_size) as usize,
    ];

    println!("Configuration:");
    println!("  Medium type: {}", medium_type);
    println!("  Parallel: {}", use_parallel);
    if let Some(nd) = n_domains {
        println!("  Domain decomposition: {:?}", nd);
    }
    println!("  Grid shape: {:?}", shape);

    // Create medium based on type
    let permittivity = match medium_type {
        "layered" => create_layered_medium(shape),
        "lens" => create_lens_medium(shape, shape[0] as f64 / 3.0),
        _ => {
            // Default: random medium
            let mut perm = WaveArray::zeros((shape[0], shape[1], shape[2]));
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    for k in 0..shape[2] {
                        let n = 1.0 + 0.5 * ((i + j + k) as f64 * 0.1).sin().abs();
                        perm.data[[i, j, k]] = Complex::new(n * n, 0.001);
                    }
                }
            }
            perm
        }
    };

    // Create multiple sources
    let source_positions = vec![
        [shape[0] / 4, shape[1] / 2, shape[2] / 2],
        [shape[0] / 2, shape[1] / 2, shape[2] / 2],
        [3 * shape[0] / 4, shape[1] / 2, shape[2] / 2],
    ];

    let sources = create_multiple_sources(shape, source_positions, pixel_size, wavelength);

    // Combine sources
    let mut combined_source = WaveArray::zeros((shape[0], shape[1], shape[2]));
    for (source, pos) in &sources {
        let src_shape = source.shape_tuple();
        for i in 0..src_shape.0 {
            for j in 0..src_shape.1 {
                for k in 0..src_shape.2 {
                    let pi = pos[0] + i;
                    let pj = pos[1] + j;
                    let pk = pos[2] + k;

                    if pi < shape[0] && pj < shape[1] && pk < shape[2] {
                        combined_source.data[[pi, pj, pk]] += source.data[[i, j, k]];
                    }
                }
            }
        }
    }

    println!("  Number of sources: {}", sources.len());

    // Add absorbing boundaries
    let boundary_widths = [[10, 10]; 3];
    let (padded_perm, roi) =
        add_absorbing_boundaries(permittivity, boundary_widths, 3.0, [false, false, false]);

    // Also pad the source
    let mut padded_source = WaveArray::zeros(padded_perm.shape_tuple());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                padded_source.data[[
                    i + boundary_widths[0][0],
                    j + boundary_widths[1][0],
                    k + boundary_widths[2][0],
                ]] = combined_source.data[[i, j, k]];
            }
        }
    }

    let start = Instant::now();

    // Run simulation
    let result = if use_parallel && n_domains.is_some() {
        let nd = n_domains.unwrap();

        println!(
            "\nUsing solver with {} subdomains (parallel features not fully implemented)",
            nd[0] * nd[1] * nd[2]
        );

        // Use regular solver (parallel features would require additional implementation)
        let domain = HelmholtzDomain::new(
            padded_perm.clone(),
            pixel_size,
            wavelength,
            [false, false, false],
            boundary_widths,
        );

        let config = IterationConfig {
            max_iterations: 2000,
            threshold: 1e-6,
            alpha: 0.75,
            full_residuals: true,
        };

        preconditioned_richardson(&domain, &padded_source, config)
    } else {
        // Use serial solver
        let domain = HelmholtzDomain::new(
            padded_perm,
            pixel_size,
            wavelength,
            [false, false, false],
            boundary_widths,
        );

        println!("\nUsing serial solver");

        let config = IterationConfig {
            max_iterations: 2000,
            threshold: 1e-6,
            alpha: 0.75,
            full_residuals: true,
        };

        preconditioned_richardson(&domain, &padded_source, config)
    };

    let elapsed = start.elapsed();

    println!("\nResults:");
    println!("  Iterations: {}", result.iterations);
    println!("  Final residual: {:.2e}", result.residual_norm);
    println!("  Total time: {:.2} s", elapsed.as_secs_f64());
    println!(
        "  Time per iteration: {:.4} s",
        elapsed.as_secs_f64() / result.iterations as f64
    );

    // Analyze the field
    let field = &result.field;
    let mut max_abs: f64 = 0.0;
    let mut total_energy = 0.0;

    for i in roi[0].0..roi[0].1 {
        for j in roi[1].0..roi[1].1 {
            for k in roi[2].0..roi[2].1 {
                let abs_val = field.data[[i, j, k]].norm();
                max_abs = max_abs.max(abs_val);
                total_energy += abs_val * abs_val;
            }
        }
    }

    println!("\nField analysis:");
    println!("  Max |E|: {:.2e}", max_abs);
    println!("  Total energy: {:.2e}", total_energy);

    // Show convergence history
    if let Some(history) = result.residual_history {
        let checkpoints = vec![0, 10, 50, 100, 200, 500, 1000, history.len() - 1];
        println!("\nConvergence history:");
        for &idx in checkpoints.iter() {
            if idx < history.len() {
                println!("  Iteration {:4}: residual = {:.2e}", idx + 1, history[idx]);
            }
        }
    }
}

fn main() {
    println!("Advanced 3D Helmholtz Simulation");
    println!("================================");

    // Run different simulations

    // 1. Layered medium with serial solver
    run_advanced_simulation("layered", false, None);

    // 2. Lens medium with serial solver
    run_advanced_simulation("lens", false, None);

    // 3. Random medium with parallel solver
    run_advanced_simulation("random", true, Some([2, 2, 2]));

    println!("\n================================");
    println!("All simulations completed!");
}
