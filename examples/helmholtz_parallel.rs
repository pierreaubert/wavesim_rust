//! Parallel Helmholtz equation solver example with domain decomposition
//!
//! Demonstrates solving the Helmholtz equation using domain decomposition

use num_complex::Complex;
use std::env;
use std::time::Instant;
use wavesim::domain::helmholtz::HelmholtzDomain;
use wavesim::domain::helmholtz_schwarz::solve_helmholtz_schwarz;
use wavesim::domain::simulation::{simulate, SimulationParams, Source};
use wavesim::engine::array::{Complex64, WaveArray};

fn main() {
    println!("Parallel Helmholtz Equation Solver Example");
    println!("===========================================\n");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let run_serial = args.iter().any(|arg| arg == "--serial");
    let run_parallel = args.iter().any(|arg| arg == "--parallel") || !run_serial;

    // Parse subdomain configuration
    let subdomains = if let Some(pos) = args.iter().position(|arg| arg == "--subdomains") {
        if pos + 3 < args.len() {
            let nx: usize = args[pos + 1].parse().unwrap_or(2);
            let ny: usize = args[pos + 2].parse().unwrap_or(2);
            let nz: usize = args[pos + 3].parse().unwrap_or(2);
            (nx, ny, nz)
        } else {
            (2, 2, 2)
        }
    } else {
        (2, 2, 2)
    };

    // Simulation parameters
    let shape = (64, 64, 64); // Computational domain size
    let pixel_size = 0.1; // 100 nm pixel size
    let wavelength = 0.5; // 500 nm wavelength

    println!("Configuration:");
    println!("  Grid size: {}x{}x{} voxels", shape.0, shape.1, shape.2);
    println!("  Pixel size: {} μm", pixel_size);
    println!("  Wavelength: {} μm", wavelength);
    if run_parallel {
        println!(
            "  Subdomains: {}x{}x{} = {} total",
            subdomains.0,
            subdomains.1,
            subdomains.2,
            subdomains.0 * subdomains.1 * subdomains.2
        );
    }

    // Create permittivity distribution (refractive index squared)
    let mut permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

    // Add a high refractive index sphere in the center
    let center = (shape.0 / 2, shape.1 / 2, shape.2 / 2);
    let radius = 10.0;
    let n_sphere = 1.5; // Refractive index of the sphere

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                let dx = (i as f64 - center.0 as f64) * pixel_size;
                let dy = (j as f64 - center.1 as f64) * pixel_size;
                let dz = (k as f64 - center.2 as f64) * pixel_size;
                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                if r <= radius * pixel_size {
                    permittivity.data[[i, j, k]] = Complex::new(n_sphere * n_sphere, 0.0);
                }
            }
        }
    }

    println!("\nMedium:");
    println!("  Background: vacuum (n=1.0)");
    println!("  Sphere: radius={} pixels, n={}", radius, n_sphere);

    // Create source field (plane wave)
    let mut source = WaveArray::zeros(shape);
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            source.data[[i, j, shape.2 / 2]] = Complex::new(1.0, 0.0);
        }
    }

    println!("\nSource:");
    println!("  Type: Plane wave");
    println!("  Position: z = {} (center)", shape.2 / 2);

    // Solver parameters
    let max_iterations = 100;
    let tolerance = 1e-6;

    println!("\nSolver parameters:");
    println!("  Max iterations: {}", max_iterations);
    println!("  Tolerance: {:.1e}\n", tolerance);

    // Run serial simulation if requested
    if run_serial {
        println!("Running SERIAL Helmholtz solver...");

        // Convert source to Source struct
        let source_struct = Source::from_data(source.clone(), [0, 0, 0]);

        let params = SimulationParams {
            wavelength,
            pixel_size,
            boundary_width: 0.2, // 200 nm
            periodic: [false, false, false],
            max_iterations,
            threshold: tolerance,
            alpha: 0.75,
            full_residuals: false,
            crop_boundaries: false,
            n_domains: None,
        };

        let start = Instant::now();

        let result = simulate(permittivity.data.clone(), vec![source_struct], params);

        let duration = start.elapsed();

        println!("\nSerial simulation completed:");
        println!("  Time: {:.3} s", duration.as_secs_f64());
        println!("  Iterations: {}", result.iterations);
        println!("  Final residual: {:.2e}", result.residual_norm);

        // Analyze solution
        analyze_solution(&result.field, shape);
    }

    // Run parallel simulation if requested
    if run_parallel {
        println!("\nRunning PARALLEL Helmholtz solver (Schwarz method)...");
        println!(
            "  Using {} subdomains",
            subdomains.0 * subdomains.1 * subdomains.2
        );

        // Create global domain
        let global_domain = HelmholtzDomain::new(
            permittivity.clone(),
            pixel_size,
            wavelength,
            [false, false, false],    // Non-periodic boundaries
            [[2, 2], [2, 2], [2, 2]], // Boundary widths
        );

        // Set number of Rayon threads (optional)
        if let Ok(threads) = env::var("RAYON_NUM_THREADS") {
            println!("  Using {} Rayon threads", threads);
        } else {
            println!("  Using default Rayon thread pool");
        }

        let start = Instant::now();

        let solution = solve_helmholtz_schwarz(
            global_domain,
            source.clone(),
            subdomains,
            max_iterations,
            tolerance,
        );

        let duration = start.elapsed();

        println!("\nParallel simulation completed:");
        println!("  Time: {:.3} s", duration.as_secs_f64());

        // Analyze solution
        analyze_solution(&solution, shape);
    }

    println!("\n✓ Helmholtz simulation successful!");

    // Print usage information
    if args.len() == 1 {
        println!("\nUsage:");
        println!("  cargo run --release --example helmholtz_parallel [options]");
        println!("\nOptions:");
        println!("  --serial              Run serial simulation only");
        println!("  --parallel            Run parallel simulation only (default)");
        println!("  --subdomains NX NY NZ Set subdomain decomposition (default: 2 2 2)");
        println!("\nEnvironment variables:");
        println!("  RAYON_NUM_THREADS=N   Set number of threads for parallel execution");
        println!("\nExamples:");
        println!("  cargo run --release --example helmholtz_parallel --serial");
        println!("  cargo run --release --example helmholtz_parallel --subdomains 4 4 4");
        println!("  RAYON_NUM_THREADS=8 cargo run --release --example helmholtz_parallel");
    }
}

/// Analyze the solution field
fn analyze_solution(solution: &WaveArray<Complex64>, shape: (usize, usize, usize)) {
    // Calculate field statistics
    let mut max_amplitude: f64 = 0.0;
    let mut total_intensity: f64 = 0.0;

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                let amplitude = solution.data[[i, j, k]].norm();
                let intensity = amplitude * amplitude;

                max_amplitude = max_amplitude.max(amplitude);
                total_intensity += intensity;
            }
        }
    }

    let avg_intensity = total_intensity / (shape.0 * shape.1 * shape.2) as f64;

    println!("\nField statistics:");
    println!("  Max amplitude: {:.3e}", max_amplitude);
    println!("  Average intensity: {:.3e}", avg_intensity);
    println!("  Total energy: {:.3e}", total_intensity);

    // Check field at specific points
    let center = (shape.0 / 2, shape.1 / 2, shape.2 / 2);
    let center_field = solution.data[[center.0, center.1, center.2]];

    println!("\nField at center:");
    println!("  Amplitude: {:.3e}", center_field.norm());
    println!("  Phase: {:.3} rad", center_field.arg());

    // Calculate scattering efficiency
    let incident_intensity = 1.0; // Unit plane wave
    let scattered_power = total_intensity - incident_intensity * (shape.0 * shape.1) as f64;

    println!("\nScattering analysis:");
    println!("  Scattered power: {:.3e}", scattered_power.abs());
    println!(
        "  Scattering efficiency: {:.1}%",
        100.0 * scattered_power.abs() / (incident_intensity * (shape.0 * shape.1) as f64)
    );
}
