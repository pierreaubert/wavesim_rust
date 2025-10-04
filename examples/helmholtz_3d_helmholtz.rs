//! 3D Helmholtz Equation in Spherical Domain
//!
//! This example demonstrates solving the 3D Helmholtz equation in a spherical domain
//! with a point source at the center. This is a well-known analytical case that allows
//! for validation of the numerical solver.
//!
//! The example shows:
//! - Numerical solution of the Helmholtz equation in 3D
//! - Comparison with analytical solutions (Green's function for sphere)
//! - Visualization of 3D fields through 2D slices
//! - Error analysis between numerical and analytical solutions

use num_complex::Complex;
use plotly::{
    common::{ColorScale, ColorScalePalette, Title},
    layout::{Axis, Layout},
    HeatMap, Plot, Scatter,
};
use std::f64::consts::PI;
use wavesim::{
    domain::helmholtz::HelmholtzDomain,
    domain::iteration::{preconditioned_richardson, IterationConfig},
    engine::array::{Complex64, WaveArray},
    utilities::analytical::{
        compare_solutions, BoundaryCondition, SphereParams, SphericalSolution,
    },
};

fn main() {
    println!("3D Helmholtz Equation in Spherical Domain");
    println!("==========================================\n");

    // Create output directory for plots
    std::fs::create_dir_all("plots/analytical").unwrap();

    // Domain parameters
    let radius = 1.0; // meters
    let freq = 500.0; // Hz
    let c_sound = 343.0; // m/s
    let wavelength = c_sound / freq;

    println!("Domain Configuration:");
    println!("  Geometry: Spherical");
    println!("  Radius: {:.2} m", radius);
    println!("  Frequency: {:.0} Hz", freq);
    println!("  Wavelength: {:.3} m", wavelength);
    println!("  Source: Point source at center (0, 0, 0)\n");

    // Grid parameters
    let grid_size = 64;
    let shape = (grid_size, grid_size, grid_size);
    let domain_size = 2.2 * radius; // Add some margin
    let pixel_size = domain_size / grid_size as f64;

    println!("Grid Parameters:");
    println!("  Grid size: {}x{}x{}", grid_size, grid_size, grid_size);
    println!("  Domain size: {:.2} m", domain_size);
    println!(
        "  Pixel size: {:.3} m ({:.1} cm)",
        pixel_size,
        pixel_size * 100.0
    );
    println!("  Points per wavelength: {:.1}\n", wavelength / pixel_size);

    // Create spherical permittivity distribution
    println!("Creating spherical medium...");
    let permittivity = create_spherical_medium(shape, pixel_size, radius, domain_size);

    // Create point source at center
    let mut source = WaveArray::zeros(shape);
    source.data[[grid_size / 2, grid_size / 2, grid_size / 2]] = Complex64::new(1.0, 0.0);

    // Set up numerical solver
    println!("Setting up Helmholtz domain...");
    let domain = HelmholtzDomain::new(
        permittivity,
        pixel_size,
        wavelength,
        [false, false, false],    // Non-periodic boundaries
        [[4, 4], [4, 4], [4, 4]], // PML boundaries
    );

    // Solve numerically
    println!("\nRunning numerical solver...");
    let config = IterationConfig {
        max_iterations: 500,
        threshold: 1e-6,
        alpha: 0.75,
        full_residuals: false,
    };

    let result = preconditioned_richardson(&domain, &source, config);
    println!("  Converged in {} iterations", result.iterations);
    println!("  Final residual: {:.2e}", result.residual_norm);

    // Compute analytical solution
    println!("\nComputing analytical solution...");
    let analytical_field =
        compute_analytical_spherical_solution(shape, pixel_size, domain_size, radius, wavelength);

    // Compare solutions
    println!("\nComparing numerical and analytical solutions...");
    let (l2_error, max_error, rel_error) = compare_solutions(&result.field, &analytical_field);

    println!("  L2 error: {:.2e}", l2_error);
    println!("  Max error: {:.2e}", max_error);
    println!(
        "  Relative L2 error: {:.2e} ({:.2}%)",
        rel_error,
        rel_error * 100.0
    );

    // Analyze field at specific points
    println!("\nField Analysis:");
    let center_idx = grid_size / 2;
    let numerical_center = result.field.data[[center_idx, center_idx, center_idx]];
    let analytical_center = analytical_field.data[[center_idx, center_idx, center_idx]];

    println!("  At center (0, 0, 0):");
    println!("    Numerical: {:.4e}", numerical_center.norm());
    println!("    Analytical: {:.4e}", analytical_center.norm());
    println!(
        "    Error: {:.2e}",
        (numerical_center - analytical_center).norm()
    );

    // Point at radius/2 along x-axis
    let mid_radius_idx = center_idx + (radius / (2.0 * pixel_size)) as usize;
    if mid_radius_idx < grid_size {
        let numerical_mid = result.field.data[[mid_radius_idx, center_idx, center_idx]];
        let analytical_mid = analytical_field.data[[mid_radius_idx, center_idx, center_idx]];

        println!("  At ({:.2}, 0, 0):", radius / 2.0);
        println!("    Numerical: {:.4e}", numerical_mid.norm());
        println!("    Analytical: {:.4e}", analytical_mid.norm());
        println!("    Error: {:.2e}", (numerical_mid - analytical_mid).norm());
    }

    // Generate visualizations
    println!("\nGenerating visualizations...");

    // XY slice at z=0 (center)
    plot_2d_slice(
        &result.field,
        center_idx,
        pixel_size,
        domain_size,
        "Numerical Solution - XY Slice (z=0)",
        "plots/analytical/sphere_3d_numerical_xy",
    );

    plot_2d_slice(
        &analytical_field,
        center_idx,
        pixel_size,
        domain_size,
        "Analytical Solution - XY Slice (z=0)",
        "plots/analytical/sphere_3d_analytical_xy",
    );

    // Error field
    let mut error_field = result.field.clone();
    for (num_val, ana_val) in error_field
        .data
        .iter_mut()
        .zip(analytical_field.data.iter())
    {
        *num_val = *num_val - *ana_val;
    }

    plot_2d_slice(
        &error_field,
        center_idx,
        pixel_size,
        domain_size,
        "Error Field (Numerical - Analytical) - XY Slice (z=0)",
        "plots/analytical/sphere_3d_error_xy",
    );

    // XZ slice at y=0
    plot_xz_slice(
        &result.field,
        center_idx,
        pixel_size,
        domain_size,
        "Numerical Solution - XZ Slice (y=0)",
        "plots/analytical/sphere_3d_numerical_xz",
    );

    plot_xz_slice(
        &analytical_field,
        center_idx,
        pixel_size,
        domain_size,
        "Analytical Solution - XZ Slice (y=0)",
        "plots/analytical/sphere_3d_analytical_xz",
    );

    // Radial profile along x-axis
    plot_radial_profile(
        &result.field,
        &analytical_field,
        grid_size,
        pixel_size,
        domain_size,
    );

    println!("\n✓ All visualizations completed!");
    println!("Check the 'plots/analytical' directory for output files.");
    println!("\nGenerated plots:");
    println!("  - plots/analytical/sphere_3d_numerical_xy.html");
    println!("  - plots/analytical/sphere_3d_analytical_xy.html");
    println!("  - plots/analytical/sphere_3d_error_xy.html");
    println!("  - plots/analytical/sphere_3d_numerical_xz.html");
    println!("  - plots/analytical/sphere_3d_analytical_xz.html");
    println!("  - plots/analytical/sphere_3d_radial_profile.html");
}

/// Create a spherical medium with given radius
fn create_spherical_medium(
    shape: (usize, usize, usize),
    pixel_size: f64,
    radius: f64,
    domain_size: f64,
) -> WaveArray<Complex64> {
    let mut medium = WaveArray::zeros(shape);
    let center = shape.0 as f64 / 2.0;

    let n_air = 1.0;
    let n_boundary = Complex::new(1.0, 0.5); // Absorbing boundary

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                // Calculate distance from center
                let x = (i as f64 - center) * pixel_size;
                let y = (j as f64 - center) * pixel_size;
                let z = (k as f64 - center) * pixel_size;
                let r = (x * x + y * y + z * z).sqrt();

                // Inside sphere: air (n=1), outside: absorbing
                if r <= radius {
                    medium.data[[i, j, k]] = Complex64::new(n_air * n_air, 0.0);
                } else {
                    medium.data[[i, j, k]] = n_boundary * n_boundary;
                }
            }
        }
    }

    medium
}

/// Compute analytical solution for spherical domain with point source at center
fn compute_analytical_spherical_solution(
    shape: (usize, usize, usize),
    pixel_size: f64,
    domain_size: f64,
    radius: f64,
    wavelength: f64,
) -> WaveArray<Complex64> {
    let mut field = WaveArray::zeros(shape);
    let center = shape.0 as f64 / 2.0;
    let k = 2.0 * PI / wavelength;

    // For a point source at the origin, the Green's function is:
    // G(r) = exp(i*k*r) / (4*π*r)
    // Inside a sphere with appropriate boundary conditions

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k_idx in 0..shape.2 {
                let x = (i as f64 - center) * pixel_size;
                let y = (j as f64 - center) * pixel_size;
                let z = (k_idx as f64 - center) * pixel_size;
                let r = (x * x + y * y + z * z).sqrt();

                // Avoid singularity at origin
                if r < pixel_size / 2.0 {
                    field.data[[i, j, k_idx]] = Complex64::new(0.0, 0.0);
                } else if r <= radius {
                    // Green's function inside sphere
                    let phase = Complex64::new(0.0, k * r).exp();
                    field.data[[i, j, k_idx]] = phase / (4.0 * PI * r);
                } else {
                    // Outside sphere: exponential decay
                    let decay = (-2.0 * (r - radius)).exp();
                    let phase = Complex64::new(0.0, k * r).exp();
                    field.data[[i, j, k_idx]] = phase * decay / (4.0 * PI * r);
                }
            }
        }
    }

    field
}

/// Plot a 2D slice from a 3D field (XY plane at given z)
fn plot_2d_slice(
    field: &WaveArray<Complex64>,
    slice_z: usize,
    pixel_size: f64,
    domain_size: f64,
    title: &str,
    filename: &str,
) {
    let shape = field.shape_tuple();
    let (nx, ny, _nz) = shape;

    let mut magnitude = vec![];
    for j in 0..ny {
        let mut row = vec![];
        for i in 0..nx {
            let value = field.data[[i, j, slice_z]].norm();
            row.push(value);
        }
        magnitude.push(row);
    }

    // Create coordinate arrays (centered at 0)
    let x_coords: Vec<f64> = (0..nx)
        .map(|i| (i as f64 / nx as f64 - 0.5) * domain_size)
        .collect();
    let y_coords: Vec<f64> = (0..ny)
        .map(|j| (j as f64 / ny as f64 - 0.5) * domain_size)
        .collect();

    let trace = HeatMap::new_z(magnitude)
        .x(x_coords)
        .y(y_coords)
        .color_scale(ColorScale::Palette(ColorScalePalette::Viridis));

    let layout = Layout::new()
        .title(Title::from(title))
        .x_axis(Axis::new().title("X (m)"))
        .y_axis(Axis::new().title("Y (m)"))
        .width(700)
        .height(700);

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);

    let html_path = format!("{}.html", filename);
    plot.write_html(&html_path);
    println!("  Saved plot: {}", html_path);
}

/// Plot a 2D slice from a 3D field (XZ plane at given y)
fn plot_xz_slice(
    field: &WaveArray<Complex64>,
    slice_y: usize,
    pixel_size: f64,
    domain_size: f64,
    title: &str,
    filename: &str,
) {
    let shape = field.shape_tuple();
    let (nx, _ny, nz) = shape;

    let mut magnitude = vec![];
    for k in 0..nz {
        let mut row = vec![];
        for i in 0..nx {
            let value = field.data[[i, slice_y, k]].norm();
            row.push(value);
        }
        magnitude.push(row);
    }

    let x_coords: Vec<f64> = (0..nx)
        .map(|i| (i as f64 / nx as f64 - 0.5) * domain_size)
        .collect();
    let z_coords: Vec<f64> = (0..nz)
        .map(|k| (k as f64 / nz as f64 - 0.5) * domain_size)
        .collect();

    let trace = HeatMap::new_z(magnitude)
        .x(x_coords)
        .y(z_coords)
        .color_scale(ColorScale::Palette(ColorScalePalette::Viridis));

    let layout = Layout::new()
        .title(Title::from(title))
        .x_axis(Axis::new().title("X (m)"))
        .y_axis(Axis::new().title("Z (m)"))
        .width(700)
        .height(700);

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);

    let html_path = format!("{}.html", filename);
    plot.write_html(&html_path);
    println!("  Saved plot: {}", html_path);
}

/// Plot radial profile along x-axis
fn plot_radial_profile(
    numerical_field: &WaveArray<Complex64>,
    analytical_field: &WaveArray<Complex64>,
    grid_size: usize,
    pixel_size: f64,
    domain_size: f64,
) {
    let center = grid_size / 2;

    let mut radii = vec![];
    let mut numerical_values = vec![];
    let mut analytical_values = vec![];

    // Extract values along x-axis (y=center, z=center)
    for i in 0..grid_size {
        let r = ((i as f64 - center as f64) * pixel_size).abs();
        radii.push(r);
        numerical_values.push(numerical_field.data[[i, center, center]].norm());
        analytical_values.push(analytical_field.data[[i, center, center]].norm());
    }

    // Create traces
    let numerical_trace = Scatter::new(radii.clone(), numerical_values)
        .mode(plotly::common::Mode::Lines)
        .name("Numerical")
        .line(plotly::common::Line::new().color("blue").width(2.0));

    let analytical_trace = Scatter::new(radii, analytical_values)
        .mode(plotly::common::Mode::Lines)
        .name("Analytical")
        .line(
            plotly::common::Line::new()
                .color("red")
                .width(2.0)
                .dash(plotly::common::DashType::Dash),
        );

    let layout = Layout::new()
        .title(Title::from("Radial Profile Along X-Axis"))
        .x_axis(Axis::new().title("Radial Distance (m)"))
        .y_axis(Axis::new().title("Field Magnitude"))
        .width(800)
        .height(600);

    let mut plot = Plot::new();
    plot.add_trace(numerical_trace);
    plot.add_trace(analytical_trace);
    plot.set_layout(layout);

    let filename = "plots/analytical/sphere_3d_radial_profile.html";
    plot.write_html(filename);
    println!("  Saved plot: {}", filename);
}
