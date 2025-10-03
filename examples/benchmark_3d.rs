//! Benchmark example for comparing Rust vs Python performance
//!
//! This example runs a standardized 3D Helmholtz simulation
//! that can be directly compared with the Python implementation.

use num_complex::Complex;
use std::time::Instant;
use wavesim::{
    domain::domain_trait::Domain,
    domain::helmholtz::HelmholtzDomain,
    domain::iteration::{preconditioned_richardson, IterationConfig},
    engine::array::WaveArray,
};

type Complex64 = Complex<f64>;

/// Create a standardized test medium with known properties
fn create_benchmark_medium(shape: (usize, usize, usize)) -> WaveArray<Complex64> {
    let mut permittivity = WaveArray::zeros(shape);

    let center_x = shape.0 / 2;
    let center_y = shape.1 / 2;
    let center_z = shape.2 / 2;

    // Create a sphere of higher refractive index in the center
    let radius = shape.0.min(shape.1).min(shape.2) as f64 / 4.0;

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                let dx = i as f64 - center_x as f64;
                let dy = j as f64 - center_y as f64;
                let dz = k as f64 - center_z as f64;

                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                let n = if r <= radius {
                    1.5 // Higher refractive index sphere
                } else {
                    1.0 // Background
                };

                // Small absorption for stability
                permittivity.data[[i, j, k]] = Complex::new(n * n, 0.001);
            }
        }
    }

    permittivity
}

fn run_benchmark(size: usize, max_iterations: usize, print_details: bool) -> (f64, usize, f64) {
    let shape = (size, size, size);

    if print_details {
        println!("\n=== Benchmark: {}x{}x{} grid ===", size, size, size);
        println!("Total voxels: {}", size * size * size);
    }

    // Fixed parameters for reproducibility
    let wavelength = 1.0;
    let pixel_size = 0.25;

    // Create medium
    let permittivity = create_benchmark_medium(shape);

    // Create point source at one corner
    let mut source = WaveArray::zeros(shape);
    source.data[[1, 1, 1]] = Complex::new(1.0, 0.0);

    // Create domain
    let domain = HelmholtzDomain::new(
        permittivity,
        pixel_size,
        wavelength,
        [true, true, true], // Periodic for simplicity
        [[0, 0], [0, 0], [0, 0]],
    );

    // Run simulation with timing
    let start = Instant::now();

    let config = IterationConfig {
        max_iterations,
        threshold: 1e-6,
        alpha: 0.75,
        full_residuals: false,
    };

    let result = preconditioned_richardson(&domain, &source, config);

    let elapsed = start.elapsed().as_secs_f64();

    if print_details {
        println!("Time: {:.3} s", elapsed);
        println!("Iterations: {}", result.iterations);
        println!(
            "Time per iteration: {:.4} s",
            elapsed / result.iterations as f64
        );
        println!("Final residual: {:.2e}", result.residual_norm);

        // Calculate throughput metrics
        let mvoxels = (size * size * size) as f64 / 1e6;
        let mvoxels_per_sec = mvoxels * result.iterations as f64 / elapsed;
        println!("Throughput: {:.1} Mvoxel-iterations/s", mvoxels_per_sec);
    }

    (elapsed, result.iterations, result.residual_norm)
}

fn main() {
    println!("Helmholtz 3D Performance Benchmark (Rust)");
    println!("==========================================");

    // Warm-up run
    println!("\nWarm-up run...");
    run_benchmark(16, 10, false);

    // Benchmark different sizes
    let sizes = vec![16, 24, 32, 48, 64];
    let max_iters = 100;

    println!("\nRunning benchmarks...");
    println!("\nSize    | Voxels    | Time (s) | Iters | ms/iter | Mvox-it/s | Residual");
    println!("--------|-----------|----------|-------|---------|-----------|----------");

    for size in sizes {
        let (time, iters, residual) = run_benchmark(size, max_iters, false);
        let voxels = size * size * size;
        let ms_per_iter = (time * 1000.0) / iters as f64;
        let mvoxel_iters_per_sec = (voxels as f64 * iters as f64) / (time * 1e6);

        println!(
            "{:7} | {:9} | {:8.3} | {:5} | {:7.2} | {:9.1} | {:.2e}",
            format!("{}³", size),
            voxels,
            time,
            iters,
            ms_per_iter,
            mvoxel_iters_per_sec,
            residual
        );
    }

    // Detailed run for profiling
    println!("\n\nDetailed benchmark (32x32x32):");
    println!("--------------------------------");
    let iterations_test = vec![10, 50, 100, 200];

    for max_iter in iterations_test {
        let (time, iters, _) = run_benchmark(32, max_iter, false);
        println!(
            "  {:3} iterations: {:.3} s ({:.2} ms/iter)",
            iters,
            time,
            (time * 1000.0) / iters as f64
        );
    }

    // Memory footprint estimate
    println!("\nMemory usage estimate:");
    for size in vec![32, 64, 128] {
        let voxels = size * size * size;
        let bytes_per_complex = 16; // 8 bytes real + 8 bytes imag
        let arrays_needed = 10; // Rough estimate for solver workspace
        let memory_mb = (voxels * bytes_per_complex * arrays_needed) as f64 / (1024.0 * 1024.0);
        println!("  {}³ grid: ~{:.1} MB", size, memory_mb);
    }

    println!("\nBenchmark complete!");
}
