//! Simple Helmholtz 3D example with basic plotly visualization
//!
//! This example demonstrates solving the 3D Helmholtz equation
//! and visualizing the results using plotly.

use num_complex::Complex;
use plotly::{
    common::{ColorScale, ColorScalePalette, Title},
    HeatMap, Layout, Plot, Scatter,
};
use std::f64::consts::PI;
use wavesim::{
    domain::helmholtz::HelmholtzDomain,
    domain::iteration::{preconditioned_richardson, IterationConfig},
    engine::array::WaveArray,
};

type Complex64 = Complex<f64>;

/// Save 2D slice as heatmap (simplified version)
fn plot_2d_slice(
    field: &WaveArray<Complex64>,
    slice_z: usize,
    pixel_size: f64,
    title: &str,
    filename: &str,
) {
    let shape = field.shape_tuple();
    let (nx, ny, _nz) = shape;

    // Extract magnitude at z = slice_z
    let mut magnitude = vec![];
    for i in 0..nx {
        let mut row = vec![];
        for j in 0..ny {
            let value = field.data[[i, j, slice_z]].norm();
            row.push(value);
        }
        magnitude.push(row);
    }

    // Create axis labels
    let x_labels: Vec<String> = (0..nx)
        .step_by(nx.max(1) / 10) // Sample labels to avoid crowding
        .map(|i| format!("{:.1}", i as f64 * pixel_size))
        .collect();

    let y_labels: Vec<String> = (0..ny)
        .step_by(ny.max(1) / 10)
        .map(|j| format!("{:.1}", j as f64 * pixel_size))
        .collect();

    let trace = HeatMap::new(x_labels, y_labels, magnitude)
        .color_scale(ColorScale::Palette(ColorScalePalette::Viridis));

    let layout = Layout::new()
        .title(Title::from(title))
        .x_axis(plotly::layout::Axis::new().title("x (μm)"))
        .y_axis(plotly::layout::Axis::new().title("y (μm)"));

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);

    // Save as HTML
    let html_path = format!("{}.html", filename);
    plot.write_html(&html_path);
    println!("Saved plot to {}", html_path);
}

/// Plot 1D line profile
fn plot_line_profile(
    field: &WaveArray<Complex64>,
    y_index: usize,
    z_index: usize,
    pixel_size: f64,
    title: &str,
    filename: &str,
) {
    let shape = field.shape_tuple();
    let (nx, _ny, _nz) = shape;

    let mut x_vals = vec![];
    let mut intensity = vec![];

    for i in 0..nx {
        x_vals.push(i as f64 * pixel_size);
        intensity.push(field.data[[i, y_index, z_index]].norm());
    }

    let trace = Scatter::new(x_vals, intensity)
        .mode(plotly::common::Mode::Lines)
        .name("Field intensity");

    let layout = Layout::new()
        .title(Title::from(title))
        .x_axis(plotly::layout::Axis::new().title("x (μm)"))
        .y_axis(plotly::layout::Axis::new().title("|E|"));

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);

    let html_path = format!("{}.html", filename);
    plot.write_html(&html_path);
    println!("Saved line profile to {}", html_path);
}

fn main() {
    println!("Helmholtz 3D Simulation with Plotly Visualization");
    println!("=================================================\n");

    // Simulation parameters (smaller for quick demo)
    let grid_size = 32;
    let shape = (grid_size, grid_size, grid_size);
    let wavelength = 0.5; // μm
    let pixel_size = 0.1; // μm

    println!("Setting up simulation...");
    println!("  Grid: {}x{}x{}", shape.0, shape.1, shape.2);
    println!("  Wavelength: {} μm", wavelength);
    println!("  Pixel size: {} μm", pixel_size);

    // Create simple medium with a high-index sphere
    let mut permittivity = WaveArray::zeros(shape);
    let center = grid_size / 2;
    let radius = grid_size as f64 / 6.0;

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                let dx = i as f64 - center as f64;
                let dy = j as f64 - center as f64;
                let dz = k as f64 - center as f64;
                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                let n = if r <= radius { 1.5 } else { 1.0 };
                permittivity.data[[i, j, k]] = Complex::new(n * n, 0.001);
            }
        }
    }

    // Create point source
    let mut source = WaveArray::zeros(shape);
    source.data[[1, center, center]] = Complex::new(1.0, 0.0);

    // Create domain
    let domain = HelmholtzDomain::new(
        permittivity.clone(),
        pixel_size,
        wavelength,
        [false, false, false],
        [[4, 4], [4, 4], [4, 4]],
    );

    // Solve
    println!("\nSolving Helmholtz equation...");
    let config = IterationConfig {
        max_iterations: 200,
        threshold: 1e-5,
        alpha: 0.75,
        full_residuals: false,
    };

    let result = preconditioned_richardson(&domain, &source, config);
    println!("  Converged in {} iterations", result.iterations);
    println!("  Final residual: {:.2e}", result.residual_norm);

    // Create plots directory
    std::fs::create_dir_all("plots").unwrap_or(());

    // Generate visualizations
    println!("\nGenerating plots...");

    // Plot XY slice at z=center
    plot_2d_slice(
        &result.field,
        center,
        pixel_size,
        "Field Magnitude - XY plane (z=center)",
        "plots/helmholtz_xy",
    );

    // Plot line profile along x-axis
    plot_line_profile(
        &result.field,
        center,
        center,
        pixel_size,
        "Field Profile Along X-axis",
        "plots/helmholtz_profile",
    );

    // Plot the medium structure
    plot_2d_slice(
        &permittivity,
        center,
        pixel_size,
        "Refractive Index Structure",
        "plots/helmholtz_medium",
    );

    println!("\nVisualization complete!");
    println!("Open the HTML files in 'plots/' directory to view interactive plots.");
}
