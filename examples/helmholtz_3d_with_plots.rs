//! Helmholtz 3D example with visualization
//!
//! This example demonstrates solving the 3D Helmholtz equation
//! and visualizing the results using plotly, similar to the Python version.

use num_complex::Complex;
use plotly::{
    common::{ColorScale, ColorScalePalette, Title},
    HeatMap, Layout, Plot, Scatter, Scatter3D,
};
use std::f64::consts::PI;
use wavesim::{
    domain::helmholtz::HelmholtzDomain,
    domain::iteration::{preconditioned_richardson, IterationConfig},
    engine::array::WaveArray,
};

type Complex64 = Complex<f64>;

/// Create a medium with a lens structure
fn create_lens_medium(
    shape: (usize, usize, usize),
    lens_radius: f64,
    n_lens: f64,
) -> WaveArray<Complex64> {
    let mut permittivity = WaveArray::zeros(shape);

    let center_x = shape.0 / 2;
    let center_y = shape.1 / 2;
    let center_z = shape.2 / 2;

    // Create a spherical lens
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                let dx = i as f64 - center_x as f64;
                let dy = j as f64 - center_y as f64;
                let dz = k as f64 - center_z as f64;

                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                let n = if r <= lens_radius {
                    n_lens // Higher refractive index lens
                } else {
                    1.0 // Background (air/vacuum)
                };

                // Small absorption for numerical stability
                permittivity.data[[i, j, k]] = Complex::new(n * n, 0.001);
            }
        }
    }

    permittivity
}

/// Create a Gaussian beam source
fn create_gaussian_source(shape: (usize, usize, usize), beam_waist: f64) -> WaveArray<Complex64> {
    let mut source = WaveArray::zeros(shape);

    let center_y = shape.1 / 2;
    let center_z = shape.2 / 2;

    // Place the source at x = 1 (near the boundary)
    let x_pos = 1;

    for j in 0..shape.1 {
        for k in 0..shape.2 {
            let dy = j as f64 - center_y as f64;
            let dz = k as f64 - center_z as f64;
            let r2 = dy * dy + dz * dz;

            let amplitude = (-r2 / (beam_waist * beam_waist)).exp();
            source.data[[x_pos, j, k]] = Complex::new(amplitude, 0.0);
        }
    }

    source
}

/// Save 2D slice of complex field as heatmap
fn plot_field_slice(
    field: &WaveArray<Complex64>,
    slice_index: usize,
    slice_axis: usize,
    pixel_size: f64,
    title: &str,
    filename: &str,
) {
    let shape = field.shape();

    // Extract the slice
    let (width, height, data) = match slice_axis {
        0 => {
            // YZ slice at x = slice_index
            let mut magnitude = vec![vec![0.0; shape[2]]; shape[1]];
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    magnitude[j][k] = field.data[[slice_index, j, k]].norm();
                }
            }
            (shape[1], shape[2], magnitude)
        }
        1 => {
            // XZ slice at y = slice_index
            let mut magnitude = vec![vec![0.0; shape[2]]; shape[0]];
            for i in 0..shape[0] {
                for k in 0..shape[2] {
                    magnitude[i][k] = field.data[[i, slice_index, k]].norm();
                }
            }
            (shape[0], shape[2], magnitude)
        }
        _ => {
            // XY slice at z = slice_index
            let mut magnitude = vec![vec![0.0; shape[1]]; shape[0]];
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    magnitude[i][j] = field.data[[i, j, slice_index]].norm();
                }
            }
            (shape[0], shape[1], magnitude)
        }
    };

    // Create axis labels with proper units
    let x_axis: Vec<String> = (0..width)
        .map(|i| format!("{:.1}", i as f64 * pixel_size))
        .collect();
    let y_axis: Vec<String> = (0..height)
        .map(|i| format!("{:.1}", i as f64 * pixel_size))
        .collect();

    let trace = HeatMap::new(x_axis, y_axis, data)
        .color_scale(ColorScale::Palette(ColorScalePalette::Viridis))
        .transpose(true)
        .name("Field magnitude");

    let layout = Layout::new()
        .title(Title::from(title))
        .x_axis(plotly::layout::Axis::new().title("x (μm)"))
        .y_axis(plotly::layout::Axis::new().title("y (μm)"))
        .height(600)
        .width(800);

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);

    // Save as HTML for interactive viewing
    plot.write_html(format!("{}.html", filename));

    println!("Saved plot to {}.html", filename);
}

/// Plot intensity profile along a line
fn plot_line_profile(
    field: &WaveArray<Complex64>,
    axis: usize,
    fixed_coords: (usize, usize),
    pixel_size: f64,
    title: &str,
    filename: &str,
) {
    let shape = field.shape();

    let (x_vals, intensity): (Vec<f64>, Vec<f64>) = match axis {
        0 => {
            // Profile along x-axis
            let x = (0..shape[0]).map(|i| i as f64 * pixel_size).collect();
            let y = (0..shape[0])
                .map(|i| field.data[[i, fixed_coords.0, fixed_coords.1]].norm())
                .collect();
            (x, y)
        }
        1 => {
            // Profile along y-axis
            let x = (0..shape[1]).map(|j| j as f64 * pixel_size).collect();
            let y = (0..shape[1])
                .map(|j| field.data[[fixed_coords.0, j, fixed_coords.1]].norm())
                .collect();
            (x, y)
        }
        _ => {
            // Profile along z-axis
            let x = (0..shape[2]).map(|k| k as f64 * pixel_size).collect();
            let y = (0..shape[2])
                .map(|k| field.data[[fixed_coords.0, fixed_coords.1, k]].norm())
                .collect();
            (x, y)
        }
    };

    let trace = Scatter::new(x_vals, intensity)
        .name("Intensity")
        .mode(plotly::common::Mode::Lines);

    let axis_name = match axis {
        0 => "x",
        1 => "y",
        _ => "z",
    };

    let layout = Layout::new()
        .title(Title::from(title))
        .x_axis(plotly::layout::Axis::new().title(format!("{} (μm)", axis_name)))
        .y_axis(plotly::layout::Axis::new().title("Field magnitude |E|"))
        .height(400)
        .width(600);

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);

    plot.write_html(format!("{}.html", filename));
    println!("Saved line profile to {}.html", filename);
}

/// Plot 3D isosurface of field magnitude
fn plot_3d_isosurface(
    field: &WaveArray<Complex64>,
    pixel_size: f64,
    threshold: f64,
    title: &str,
    filename: &str,
) {
    let shape = field.shape();

    // Create coordinate grids
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut z = Vec::new();
    let mut values = Vec::new();

    // Sample the field (reduce resolution for visualization)
    let step = 2; // Sample every 2nd point to reduce data size
    for i in (0..shape[0]).step_by(step) {
        for j in (0..shape[1]).step_by(step) {
            for k in (0..shape[2]).step_by(step) {
                let magnitude = field.data[[i, j, k]].norm();
                if magnitude > threshold {
                    x.push(i as f64 * pixel_size);
                    y.push(j as f64 * pixel_size);
                    z.push(k as f64 * pixel_size);
                    values.push(magnitude);
                }
            }
        }
    }

    // Create 3D scatter plot with points sized by magnitude
    let trace = Scatter3D::new(x, y, z)
        .mode(plotly::common::Mode::Markers)
        .name("Field magnitude");

    let layout = Layout::new()
        .title(Title::from(title))
        .height(700)
        .width(800);

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);

    plot.write_html(format!("{}.html", filename));
    println!("Saved 3D visualization to {}.html", filename);
}

fn main() {
    println!("Helmholtz 3D Simulation with Visualization");
    println!("==========================================\n");

    // Simulation parameters
    let grid_size = 64;
    let shape = (grid_size, grid_size, grid_size);
    let wavelength = 0.5; // μm (green light)
    let pixel_size = 0.1; // μm
    let lens_radius = 2.0; // μm
    let n_lens = 1.5; // Glass refractive index
    let beam_waist = 1.0; // μm

    println!("Creating medium with spherical lens...");
    println!("  Grid: {}x{}x{}", shape.0, shape.1, shape.2);
    println!("  Wavelength: {} μm", wavelength);
    println!("  Pixel size: {} μm", pixel_size);
    println!("  Lens radius: {} μm", lens_radius);
    println!("  Lens refractive index: {}", n_lens);

    // Create medium
    let permittivity = create_lens_medium(shape, lens_radius / pixel_size, n_lens);

    // Create Gaussian beam source
    println!("\nCreating Gaussian beam source...");
    println!("  Beam waist: {} μm", beam_waist);
    let source = create_gaussian_source(shape, beam_waist / pixel_size);

    // Create Helmholtz domain
    let domain = HelmholtzDomain::new(
        permittivity.clone(),
        pixel_size,
        wavelength,
        [false, false, false],    // Non-periodic boundaries
        [[8, 8], [8, 8], [8, 8]], // PML boundaries
    );

    // Solve
    println!("\nSolving Helmholtz equation...");
    let config = IterationConfig {
        max_iterations: 1000,
        threshold: 1e-6,
        alpha: 0.75,
        full_residuals: false,
    };

    let result = preconditioned_richardson(&domain, &source, config);

    println!("  Converged in {} iterations", result.iterations);
    println!("  Final residual: {:.2e}", result.residual_norm);

    // Create visualization directory
    std::fs::create_dir_all("plots").unwrap();

    // Plot results
    println!("\nGenerating visualizations...");

    // 1. Plot central XY slice (z = center)
    plot_field_slice(
        &result.field,
        shape.2 / 2,
        2,
        pixel_size,
        "Field magnitude - XY plane (z=center)",
        "plots/helmholtz_xy_slice",
    );

    // 2. Plot central XZ slice (y = center)
    plot_field_slice(
        &result.field,
        shape.1 / 2,
        1,
        pixel_size,
        "Field magnitude - XZ plane (y=center)",
        "plots/helmholtz_xz_slice",
    );

    // 3. Plot central YZ slice (x = center)
    plot_field_slice(
        &result.field,
        shape.0 / 2,
        0,
        pixel_size,
        "Field magnitude - YZ plane (x=center)",
        "plots/helmholtz_yz_slice",
    );

    // 4. Plot line profile along beam propagation (x-axis)
    plot_line_profile(
        &result.field,
        0,
        (shape.1 / 2, shape.2 / 2),
        pixel_size,
        "Field intensity along beam axis",
        "plots/helmholtz_beam_profile",
    );

    // 5. Plot 3D visualization (reduced resolution)
    let max_field = result
        .field
        .data
        .iter()
        .map(|c| c.norm())
        .fold(0.0, f64::max);

    plot_3d_isosurface(
        &result.field,
        pixel_size,
        max_field * 0.1, // Show points above 10% of max
        "3D Field Distribution",
        "plots/helmholtz_3d",
    );

    // Plot the permittivity structure
    plot_field_slice(
        &permittivity,
        shape.2 / 2,
        2,
        pixel_size,
        "Refractive index structure - XY plane",
        "plots/helmholtz_medium",
    );

    println!("\nVisualization complete! Check the 'plots' directory for results.");
    println!("Open the .html files in a web browser for interactive plots.");
}
