//! 3D Helmholtz lens simulation example
//!
//! This example simulates wave propagation through a lens structure,
//! demonstrating focusing effects and Gaussian beam sources.

use num_complex::Complex;
use std::time::Instant;
use wavesim::{
    domain::domain_trait::Domain,
    domain::helmholtz::HelmholtzDomain,
    domain::iteration::{preconditioned_richardson, IterationConfig},
    engine::array::WaveArray,
    utilities::add_absorbing_boundaries,
};

type Complex64 = Complex<f64>;

/// Create a spherical lens with specified parameters
fn create_spherical_lens(
    shape: [usize; 3],
    lens_radius_pixels: f64,
    lens_center: [f64; 3],
    n_lens: f64,
    n_background: f64,
) -> WaveArray<Complex64> {
    let mut permittivity = WaveArray::zeros((shape[0], shape[1], shape[2]));

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let dx = i as f64 - lens_center[0];
                let dy = j as f64 - lens_center[1];
                let dz = k as f64 - lens_center[2];

                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                let n = if r <= lens_radius_pixels {
                    n_lens
                } else {
                    n_background
                };

                // Permittivity = n^2
                permittivity.data[[i, j, k]] = Complex::new(n * n, 0.0);
            }
        }
    }

    permittivity
}

/// Create a cylindrical lens (infinite in one dimension)
fn create_cylindrical_lens(
    shape: [usize; 3],
    lens_radius_pixels: f64,
    lens_center: [f64; 2], // Center in YZ plane
    n_lens: f64,
    n_background: f64,
) -> WaveArray<Complex64> {
    let mut permittivity = WaveArray::zeros((shape[0], shape[1], shape[2]));

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let dy = j as f64 - lens_center[0];
                let dz = k as f64 - lens_center[1];

                let r = (dy * dy + dz * dz).sqrt();

                let n = if r <= lens_radius_pixels {
                    n_lens
                } else {
                    n_background
                };

                permittivity.data[[i, j, k]] = Complex::new(n * n, 0.0);
            }
        }
    }

    permittivity
}

/// Create a Gaussian beam source in the YZ plane
fn create_gaussian_beam_yz(
    shape: [usize; 2],    // Shape in YZ plane
    position: [usize; 3], // Position in 3D space
    pixel_size: f64,
    wavelength: f64,
    width_factor: f64, // Alpha parameter for Gaussian width
) -> (WaveArray<Complex64>, [usize; 3]) {
    let mut source = WaveArray::zeros((1, shape[0], shape[1]));

    let center_y = shape[0] as f64 / 2.0;
    let center_z = shape[1] as f64 / 2.0;

    // Gaussian profile
    for j in 0..shape[0] {
        for k in 0..shape[1] {
            let dy = (j as f64 - center_y) * pixel_size;
            let dz = (k as f64 - center_z) * pixel_size;

            let r2 = (dy * dy + dz * dz) / (width_factor * wavelength).powi(2);
            let amplitude = (-r2).exp();

            source.data[[0, j, k]] = Complex::new(amplitude, 0.0);
        }
    }

    (source, position)
}

/// Analyze the field at the focal plane
fn analyze_focal_plane(field: &WaveArray<Complex64>, focal_plane_x: usize, pixel_size: f64) {
    let shape = field.shape_tuple();

    if focal_plane_x >= shape.0 {
        println!("Focal plane index out of bounds");
        return;
    }

    // Find the maximum intensity at the focal plane
    let mut max_intensity = 0.0;
    let mut max_pos = (0, 0);
    let mut total_intensity = 0.0;

    for j in 0..shape.1 {
        for k in 0..shape.2 {
            let intensity = field.data[[focal_plane_x, j, k]].norm_sqr();
            total_intensity += intensity;

            if intensity > max_intensity {
                max_intensity = intensity;
                max_pos = (j, k);
            }
        }
    }

    println!("\nFocal plane analysis (x = {}):", focal_plane_x);
    println!("  Max intensity: {:.2e}", max_intensity);
    println!("  Max position: ({}, {})", max_pos.0, max_pos.1);
    println!("  Total intensity: {:.2e}", total_intensity);

    // Calculate FWHM (Full Width at Half Maximum)
    let half_max = max_intensity / 2.0;
    let mut fwhm_y = 0;
    let mut fwhm_z = 0;

    // FWHM in Y direction
    for j in 0..shape.1 {
        if field.data[[focal_plane_x, j, max_pos.1]].norm_sqr() >= half_max {
            fwhm_y += 1;
        }
    }

    // FWHM in Z direction
    for k in 0..shape.2 {
        if field.data[[focal_plane_x, max_pos.0, k]].norm_sqr() >= half_max {
            fwhm_z += 1;
        }
    }

    println!("  FWHM (Y): {:.2} μm", fwhm_y as f64 * pixel_size);
    println!("  FWHM (Z): {:.2} μm", fwhm_z as f64 * pixel_size);
}

fn main() {
    println!("3D Helmholtz Lens Simulation");
    println!("============================\n");

    // Parameters
    let wavelength = 1.0; // Wavelength in micrometer (μm)
    let pixel_size = wavelength / 4.0; // Pixel size in micrometer (μm)

    // Simulation domain size
    let sim_size = [10.0, 10.0, 10.0]; // μm
    let shape = [
        (sim_size[0] / pixel_size) as usize,
        (sim_size[1] / pixel_size) as usize,
        (sim_size[2] / pixel_size) as usize,
    ];

    println!("Simulation parameters:");
    println!("  Wavelength: {} μm", wavelength);
    println!("  Pixel size: {} μm", pixel_size);
    println!("  Domain size: {:?} μm", sim_size);
    println!("  Grid shape: {:?} pixels", shape);

    // Create lens
    let lens_radius = 2.0; // μm
    let lens_radius_pixels = lens_radius / pixel_size;
    let lens_center = [
        shape[0] as f64 / 2.0,
        shape[1] as f64 / 2.0,
        shape[2] as f64 / 2.0,
    ];
    let n_lens = 1.5;
    let n_background = 1.0;

    println!("\nLens parameters:");
    println!("  Type: Spherical");
    println!(
        "  Radius: {} μm ({:.1} pixels)",
        lens_radius, lens_radius_pixels
    );
    println!("  Center: {:?}", lens_center);
    println!("  Refractive index: {}", n_lens);
    println!("  Background index: {}", n_background);

    let permittivity =
        create_spherical_lens(shape, lens_radius_pixels, lens_center, n_lens, n_background);

    // Calculate expected focal length (thin lens approximation)
    let focal_length = lens_radius / (n_lens - n_background);
    println!("  Expected focal length: {:.2} μm", focal_length);

    // Create Gaussian beam source at the left edge
    let source_shape = [shape[1], shape[2]];
    let source_position = [0, 0, 0];
    let width_factor = 3.0;

    let (source, pos) = create_gaussian_beam_yz(
        source_shape,
        source_position,
        pixel_size,
        wavelength,
        width_factor,
    );

    println!("\nGaussian beam source:");
    println!("  Shape: {:?} pixels", source_shape);
    println!("  Position: {:?}", pos);
    println!("  Width factor: {}", width_factor);

    // Add absorbing boundaries
    let boundary_widths = [[5, 5]; 3];
    let (padded_perm, roi) =
        add_absorbing_boundaries(permittivity, boundary_widths, 2.0, [false, false, false]);

    // Pad the source
    let mut padded_source = WaveArray::zeros(padded_perm.shape_tuple());
    let src_shape = source.shape_tuple();
    for i in 0..src_shape.0 {
        for j in 0..src_shape.1 {
            for k in 0..src_shape.2 {
                padded_source.data[[
                    pos[0] + boundary_widths[0][0] + i,
                    pos[1] + boundary_widths[1][0] + j,
                    pos[2] + boundary_widths[2][0] + k,
                ]] = source.data[[i, j, k]];
            }
        }
    }

    // Create domain and run simulation
    let domain = HelmholtzDomain::new(
        padded_perm,
        pixel_size,
        wavelength,
        [false, false, false],
        boundary_widths,
    );

    println!("\nRunning simulation...");
    let start = Instant::now();

    let config = IterationConfig {
        max_iterations: 1500,
        threshold: 1e-6,
        alpha: 0.75,
        full_residuals: false,
    };

    let result = preconditioned_richardson(&domain, &padded_source, config);

    let elapsed = start.elapsed();

    println!("\nSimulation results:");
    println!("  Iterations: {}", result.iterations);
    println!("  Final residual: {:.2e}", result.residual_norm);
    println!("  Total time: {:.2} s", elapsed.as_secs_f64());
    println!(
        "  Time per iteration: {:.4} s",
        elapsed.as_secs_f64() / result.iterations as f64
    );

    // Extract field in ROI
    let mut field = WaveArray::zeros((
        roi[0].1 - roi[0].0,
        roi[1].1 - roi[1].0,
        roi[2].1 - roi[2].0,
    ));

    for i in 0..(roi[0].1 - roi[0].0) {
        for j in 0..(roi[1].1 - roi[1].0) {
            for k in 0..(roi[2].1 - roi[2].0) {
                field.data[[i, j, k]] =
                    result.field.data[[roi[0].0 + i, roi[1].0 + j, roi[2].0 + k]];
            }
        }
    }

    // Analyze the field at different planes
    let focal_plane_estimate =
        (shape[0] / 2 + (focal_length / pixel_size) as usize).min(shape[0] - 1);

    analyze_focal_plane(&field, shape[0] / 2, pixel_size); // At lens center
    analyze_focal_plane(&field, focal_plane_estimate, pixel_size); // At estimated focal plane
    analyze_focal_plane(&field, shape[0] - 1, pixel_size); // At exit plane

    // Check field statistics
    let mut max_abs: f64 = 0.0;
    let mut min_abs = f64::INFINITY;

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let abs_val = field.data[[i, j, k]].norm();
                max_abs = max_abs.max(abs_val);
                if abs_val > 1e-10 {
                    min_abs = min_abs.min(abs_val);
                }
            }
        }
    }

    println!("\nField statistics:");
    println!("  |E| range: [{:.2e}, {:.2e}]", min_abs, max_abs);

    println!("\nLens simulation completed successfully!");
}
