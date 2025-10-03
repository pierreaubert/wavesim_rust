//! Parallel Maxwell FDTD simulation example with performance comparison
//!
//! Demonstrates electromagnetic wave propagation using domain decomposition

use num_complex::Complex;
use std::env;
use std::time::Instant;
use wavesim::domain::maxwell::*;
use wavesim::domain::maxwell_parallel::*;
use wavesim::engine::array::WaveArray;

fn main() {
    println!("Maxwell FDTD Parallel Simulation Example");
    println!("=========================================\n");

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
    let shape = (100, 100, 100); // Larger domain for better parallelization
    let dx = 1e-6; // 1 micrometer
    let dy = 1e-6;
    let dz = 1e-6;

    println!("Configuration:");
    println!("  Grid size: {}x{}x{} cells", shape.0, shape.1, shape.2);
    println!("  Spatial resolution: {} μm", dx * 1e6);
    if run_parallel {
        println!(
            "  Subdomains: {}x{}x{} = {} total",
            subdomains.0,
            subdomains.1,
            subdomains.2,
            subdomains.0 * subdomains.1 * subdomains.2
        );
    }

    // Create material properties
    // Free space with a dielectric block in the middle
    let mut permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
    let permeability = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

    // Add dielectric block (n=2) in the middle
    for i in 40..60 {
        for j in 40..60 {
            for k in 40..60 {
                permittivity.data[[i, j, k]] = Complex::new(4.0, 0.0); // n²=4 for n=2
            }
        }
    }

    println!("\nMedium:");
    println!("  Background: vacuum (n=1)");
    println!("  Dielectric block: 20x20x20 cells (n=2)");

    // Create source - point dipole
    let source = MaxwellSource {
        source_type: MaxwellSourceType::PointDipole {
            position: [20, 50, 50],
            orientation: Orientation::Z,
            amplitude: Complex::new(1.0, 0.0),
        },
        frequency: 3e14, // 300 THz (1 μm wavelength)
    };

    println!("\nSource:");
    println!("  Type: Point dipole (Z-oriented)");
    println!("  Position: [20, 50, 50]");
    println!("  Frequency: 300 THz (λ = 1 μm)\n");

    // Run simulation parameters
    let num_steps = 200;
    let save_interval = 50;

    println!("Simulation:");
    println!("  Steps: {}", num_steps);
    println!("  Save interval: every {} steps\n", save_interval);

    // Run serial simulation if requested
    if run_serial {
        println!("Running SERIAL FDTD simulation...");

        let domain = MaxwellDomain::new(
            permittivity.clone(),
            permeability.clone(),
            dx,
            dy,
            dz,
            [false, false, false], // Absorbing boundaries
            [5, 5, 5],             // PML thickness
        );

        println!("  Time step: {:.2e} s", domain.dt);

        let start = Instant::now();

        let results =
            simulate_maxwell(domain, vec![source.clone()], num_steps, Some(save_interval));

        let duration = start.elapsed();

        println!("\nSerial simulation completed:");
        println!("  Time: {:.3} s", duration.as_secs_f64());
        println!(
            "  Throughput: {:.1} steps/s",
            num_steps as f64 / duration.as_secs_f64()
        );
        println!("  Saved {} field snapshots", results.len());

        // Analyze final field
        if let Some(final_fields) = results.last() {
            analyze_fields(final_fields, shape);
        }
    }

    // Run parallel simulation if requested
    if run_parallel {
        println!("\nRunning PARALLEL FDTD simulation...");
        println!(
            "  Using {} subdomains",
            subdomains.0 * subdomains.1 * subdomains.2
        );

        let parallel_domain = ParallelMaxwellDomain::new(
            permittivity.clone(),
            permeability.clone(),
            dx,
            dy,
            dz,
            [false, false, false], // Absorbing boundaries
            [5, 5, 5],             // PML thickness
            subdomains,
        );

        println!("  Time step: {:.2e} s", parallel_domain.dt);

        // Set number of Rayon threads (optional)
        if let Ok(threads) = env::var("RAYON_NUM_THREADS") {
            println!("  Using {} Rayon threads", threads);
        } else {
            println!("  Using default Rayon thread pool");
        }

        let start = Instant::now();

        let results = simulate_maxwell_parallel(
            parallel_domain,
            vec![source.clone()],
            num_steps,
            Some(save_interval),
        );

        let duration = start.elapsed();

        println!("\nParallel simulation completed:");
        println!("  Time: {:.3} s", duration.as_secs_f64());
        println!(
            "  Throughput: {:.1} steps/s",
            num_steps as f64 / duration.as_secs_f64()
        );
        println!("  Saved {} field snapshots", results.len());

        // Analyze final field
        if let Some(final_fields) = results.last() {
            analyze_fields(final_fields, shape);
        }
    }

    println!("\n✓ Maxwell FDTD simulation successful!");

    // Print usage information
    if args.len() == 1 {
        println!("\nUsage:");
        println!("  cargo run --release --example maxwell_fdtd_parallel [options]");
        println!("\nOptions:");
        println!("  --serial              Run serial simulation only");
        println!("  --parallel            Run parallel simulation only (default)");
        println!("  --subdomains NX NY NZ Set subdomain decomposition (default: 2 2 2)");
        println!("\nEnvironment variables:");
        println!("  RAYON_NUM_THREADS=N   Set number of threads for parallel execution");
        println!("\nExamples:");
        println!("  cargo run --release --example maxwell_fdtd_parallel --serial");
        println!("  cargo run --release --example maxwell_fdtd_parallel --subdomains 4 4 4");
        println!("  RAYON_NUM_THREADS=8 cargo run --release --example maxwell_fdtd_parallel");
    }
}

/// Analyze electromagnetic field statistics
fn analyze_fields(fields: &ElectromagneticFields, shape: (usize, usize, usize)) {
    let mut e_max: f64 = 0.0;
    let mut h_max: f64 = 0.0;

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                let e_mag = (fields.ex.data[[i, j, k]].norm_sqr()
                    + fields.ey.data[[i, j, k]].norm_sqr()
                    + fields.ez.data[[i, j, k]].norm_sqr())
                .sqrt();

                let h_mag = (fields.hx.data[[i, j, k]].norm_sqr()
                    + fields.hy.data[[i, j, k]].norm_sqr()
                    + fields.hz.data[[i, j, k]].norm_sqr())
                .sqrt();

                e_max = e_max.max(e_mag);
                h_max = h_max.max(h_mag);
            }
        }
    }

    println!("\nFinal field statistics:");
    println!("  Max |E| field: {:.3e} V/m", e_max);
    println!("  Max |H| field: {:.3e} A/m", h_max);

    // Check energy in dielectric
    let mut energy_dielectric = 0.0;
    let mut energy_vacuum = 0.0;

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                let e_energy = fields.ex.data[[i, j, k]].norm_sqr()
                    + fields.ey.data[[i, j, k]].norm_sqr()
                    + fields.ez.data[[i, j, k]].norm_sqr();

                if i >= 40 && i < 60 && j >= 40 && j < 60 && k >= 40 && k < 60 {
                    energy_dielectric += e_energy;
                } else {
                    energy_vacuum += e_energy;
                }
            }
        }
    }

    let total_energy = energy_dielectric + energy_vacuum;
    if total_energy > 0.0 {
        println!("\nEnergy distribution:");
        println!(
            "  In dielectric: {:.1}%",
            100.0 * energy_dielectric / total_energy
        );
        println!("  In vacuum: {:.1}%", 100.0 * energy_vacuum / total_energy);
    }
}
