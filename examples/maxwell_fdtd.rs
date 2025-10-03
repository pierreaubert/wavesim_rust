//! Maxwell FDTD simulation example
//!
//! Demonstrates electromagnetic wave propagation using the FDTD method

use num_complex::Complex;
use std::time::Instant;
use wavesim::domain::maxwell::*;
use wavesim::engine::array::WaveArray;

fn main() {
    println!("Maxwell FDTD Simulation Example");
    println!("================================\n");

    // Simulation parameters
    let shape = (50, 50, 50);
    let dx = 1e-6; // 1 micrometer
    let dy = 1e-6;
    let dz = 1e-6;

    println!("Configuration:");
    println!("  Grid size: {}x{}x{} cells", shape.0, shape.1, shape.2);
    println!("  Spatial resolution: {} μm", dx * 1e6);

    // Create material properties
    // Free space with a dielectric block in the middle
    let mut permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
    let mut permeability = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

    // Add dielectric block (n=2) in the middle
    for i in 20..30 {
        for j in 20..30 {
            for k in 20..30 {
                permittivity.data[[i, j, k]] = Complex::new(4.0, 0.0); // n²=4 for n=2
            }
        }
    }

    println!("\nMedium:");
    println!("  Background: vacuum (n=1)");
    println!("  Dielectric block: 10x10x10 cells (n=2)");

    // Create Maxwell domain
    let domain = MaxwellDomain::new(
        permittivity,
        permeability,
        dx,
        dy,
        dz,
        [false, false, false], // Absorbing boundaries
        [5, 5, 5],             // PML thickness
    );

    println!("  Time step: {:.2e} s", domain.dt);
    println!("  PML boundaries: 5 cells thick\n");

    // Create source - point dipole
    let source = MaxwellSource {
        source_type: MaxwellSourceType::PointDipole {
            position: [10, 25, 25],
            orientation: Orientation::Z,
            amplitude: Complex::new(1.0, 0.0),
        },
        frequency: 3e14, // 300 THz (1 μm wavelength)
    };

    println!("Source:");
    println!("  Type: Point dipole (Z-oriented)");
    println!("  Position: [10, 25, 25]");
    println!("  Frequency: 300 THz (λ = 1 μm)\n");

    // Run simulation
    let num_steps = 500;
    let save_interval = 50;

    println!("Running FDTD simulation...");
    println!("  Steps: {}", num_steps);
    println!("  Save interval: every {} steps", save_interval);

    let start = Instant::now();

    let results = simulate_maxwell(domain, vec![source], num_steps, Some(save_interval));

    let duration = start.elapsed();

    println!("\nSimulation completed in {:.2} s", duration.as_secs_f64());
    println!("  Saved {} field snapshots", results.len());

    // Analyze final field
    if let Some(final_fields) = results.last() {
        let mut e_max: f64 = 0.0;
        let mut h_max: f64 = 0.0;

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                for k in 0..shape.2 {
                    let e_mag = (final_fields.ex.data[[i, j, k]].norm_sqr()
                        + final_fields.ey.data[[i, j, k]].norm_sqr()
                        + final_fields.ez.data[[i, j, k]].norm_sqr())
                    .sqrt();

                    let h_mag = (final_fields.hx.data[[i, j, k]].norm_sqr()
                        + final_fields.hy.data[[i, j, k]].norm_sqr()
                        + final_fields.hz.data[[i, j, k]].norm_sqr())
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
                    let e_energy = final_fields.ex.data[[i, j, k]].norm_sqr()
                        + final_fields.ey.data[[i, j, k]].norm_sqr()
                        + final_fields.ez.data[[i, j, k]].norm_sqr();

                    if i >= 20 && i < 30 && j >= 20 && j < 30 && k >= 20 && k < 30 {
                        energy_dielectric += e_energy;
                    } else {
                        energy_vacuum += e_energy;
                    }
                }
            }
        }

        let total_energy = energy_dielectric + energy_vacuum;
        if total_energy > 0.0 {
            println!("\n  Energy distribution:");
            println!(
                "    In dielectric: {:.1}%",
                100.0 * energy_dielectric / total_energy
            );
            println!(
                "    In vacuum: {:.1}%",
                100.0 * energy_vacuum / total_energy
            );
        }
    }

    println!("\n✓ Maxwell FDTD simulation successful!");
}
