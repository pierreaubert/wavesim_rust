//! Maxwell 3D example with visualization
//!
//! This example demonstrates solving the 3D time-harmonic Maxwell equations
//! and visualizing the vector field results using plotly.

use num_complex::Complex;
use plotly::{
    common::{ColorScale, ColorScalePalette, Mode, Title},
    layout::{Axis, Scene},
    HeatMap, Layout, Plot, Scatter, Scatter3D,
};
use std::f64::consts::PI;
use wavesim::{
    domain::iteration::{preconditioned_richardson, IterationConfig},
    domain::maxwell::MaxwellDomain,
    engine::array::WaveArray,
};

type Complex64 = Complex<f64>;

/// Create a waveguide structure with a rectangular core
fn create_waveguide_medium(
    shape: (usize, usize, usize),
    core_size: (f64, f64),
    n_core: f64,
) -> WaveArray<Complex64> {
    let mut permittivity = WaveArray::zeros(shape);

    let center_y = shape.1 / 2;
    let center_z = shape.2 / 2;

    let half_width_y = core_size.0 / 2.0;
    let half_width_z = core_size.1 / 2.0;

    // Create rectangular waveguide core along x-axis
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                let dy = (j as f64 - center_y as f64).abs();
                let dz = (k as f64 - center_z as f64).abs();

                let n = if dy <= half_width_y && dz <= half_width_z {
                    n_core // Waveguide core
                } else {
                    1.0 // Cladding (air/vacuum)
                };

                // Small absorption for stability
                permittivity.data[[i, j, k]] = Complex::new(n * n, 0.001);
            }
        }
    }

    permittivity
}

/// Create a polarized source for waveguide excitation
fn create_polarized_source(
    shape: (usize, usize, usize),
    polarization: usize,
    mode_profile: (f64, f64),
) -> WaveArray<Complex64> {
    // Vector field: shape (3, nx, ny, nz) where first dimension is Ex, Ey, Ez
    let mut source = WaveArray::zeros((3, shape.0, shape.1, shape.2));

    let center_y = shape.1 / 2;
    let center_z = shape.2 / 2;

    // Place source at x = 1 (near input)
    let x_pos = 1;

    // Create mode profile (fundamental mode approximation)
    for j in 0..shape.1 {
        for k in 0..shape.2 {
            let dy = j as f64 - center_y as f64;
            let dz = k as f64 - center_z as f64;

            // Gaussian approximation for fundamental mode
            let amplitude = (-dy * dy / (mode_profile.0 * mode_profile.0)
                - dz * dz / (mode_profile.1 * mode_profile.1))
                .exp();

            // Set the specified polarization component
            source.data[[polarization, x_pos, j, k]] = Complex::new(amplitude, 0.0);
        }
    }

    source
}

/// Plot vector field component slice
fn plot_vector_component(
    field: &WaveArray<Complex64>,
    component: usize,
    slice_index: usize,
    slice_axis: usize,
    pixel_size: f64,
    title: &str,
    filename: &str,
) {
    let shape = (field.shape()[1], field.shape()[2], field.shape()[3]);

    // Extract the slice for the specified component
    let (width, height, data) = match slice_axis {
        0 => {
            // YZ slice at x = slice_index
            let mut magnitude = vec![vec![0.0; shape.2]; shape.1];
            for j in 0..shape.1 {
                for k in 0..shape.2 {
                    magnitude[j][k] = field.data[[component, slice_index, j, k]].norm();
                }
            }
            (shape.1, shape.2, magnitude)
        }
        1 => {
            // XZ slice at y = slice_index
            let mut magnitude = vec![vec![0.0; shape.2]; shape.0];
            for i in 0..shape.0 {
                for k in 0..shape.2 {
                    magnitude[i][k] = field.data[[component, i, slice_index, k]].norm();
                }
            }
            (shape.0, shape.2, magnitude)
        }
        _ => {
            // XY slice at z = slice_index
            let mut magnitude = vec![vec![0.0; shape.1]; shape.0];
            for i in 0..shape.0 {
                for j in 0..shape.1 {
                    magnitude[i][j] = field.data[[component, i, j, slice_index]].norm();
                }
            }
            (shape.0, shape.1, magnitude)
        }
    };

    let x_axis: Vec<String> = (0..width)
        .map(|i| format!("{:.1}", i as f64 * pixel_size))
        .collect();
    let y_axis: Vec<String> = (0..height)
        .map(|i| format!("{:.1}", i as f64 * pixel_size))
        .collect();

    let trace = HeatMap::new(x_axis, y_axis, data)
        .color_scale(ColorScale::Palette(ColorScalePalette::Plasma))
        .transpose(true)
        .name(&format!("E{}", ["x", "y", "z"][component]));

    let layout = Layout::new()
        .title(Title::from(title))
        .x_axis(Axis::new().title("Position (μm)"))
        .y_axis(Axis::new().title("Position (μm)"))
        .height(600)
        .width(800);

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);

    plot.write_html(format!("{}.html", filename));
    println!("Saved plot to {}.html", filename);
}

/// Plot total field intensity (|E|²)
fn plot_field_intensity(
    field: &WaveArray<Complex64>,
    slice_index: usize,
    slice_axis: usize,
    pixel_size: f64,
    title: &str,
    filename: &str,
) {
    let shape = (field.shape()[1], field.shape()[2], field.shape()[3]);

    // Calculate total field intensity |Ex|² + |Ey|² + |Ez|²
    let (width, height, data) = match slice_axis {
        2 => {
            // XY slice at z = slice_index
            let mut intensity = vec![vec![0.0; shape.1]; shape.0];
            for i in 0..shape.0 {
                for j in 0..shape.1 {
                    let ex = field.data[[0, i, j, slice_index]].norm();
                    let ey = field.data[[1, i, j, slice_index]].norm();
                    let ez = field.data[[2, i, j, slice_index]].norm();
                    intensity[i][j] = ex * ex + ey * ey + ez * ez;
                }
            }
            (shape.0, shape.1, intensity)
        }
        _ => panic!("Only XY slices implemented for intensity plots"),
    };

    let x_axis: Vec<String> = (0..width)
        .map(|i| format!("{:.1}", i as f64 * pixel_size))
        .collect();
    let y_axis: Vec<String> = (0..height)
        .map(|i| format!("{:.1}", i as f64 * pixel_size))
        .collect();

    let trace = HeatMap::new(x_axis, y_axis, data)
        .color_scale(ColorScale::Palette(ColorScalePalette::Hot))
        .transpose(true)
        .name("Intensity");

    let layout = Layout::new()
        .title(Title::from(title))
        .x_axis(Axis::new().title("x (μm)"))
        .y_axis(Axis::new().title("y (μm)"))
        .height(600)
        .width(800);

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);

    plot.write_html(format!("{}.html", filename));
    println!("Saved intensity plot to {}.html", filename);
}

/// Plot vector field arrows in 2D slice
fn plot_vector_field_arrows(
    field: &WaveArray<Complex64>,
    slice_index: usize,
    pixel_size: f64,
    step: usize,
    title: &str,
    filename: &str,
) {
    let shape = (field.shape()[1], field.shape()[2], field.shape()[3]);

    // Extract Ex and Ey components at z = slice_index
    let mut x_coords = Vec::new();
    let mut y_coords = Vec::new();
    let mut ex_real = Vec::new();
    let mut ey_real = Vec::new();

    for i in (0..shape.0).step_by(step) {
        for j in (0..shape.1).step_by(step) {
            x_coords.push(i as f64 * pixel_size);
            y_coords.push(j as f64 * pixel_size);
            ex_real.push(field.data[[0, i, j, slice_index]].re);
            ey_real.push(field.data[[1, i, j, slice_index]].re);
        }
    }

    // Create quiver plot using scatter with lines
    let mut plot = Plot::new();

    // Add arrow lines
    for i in 0..x_coords.len() {
        let scale = 0.3; // Arrow scale factor
        let x_end = x_coords[i] + ex_real[i] * scale;
        let y_end = y_coords[i] + ey_real[i] * scale;

        let trace = Scatter::new(vec![x_coords[i], x_end], vec![y_coords[i], y_end])
            .mode(Mode::Lines)
            .line(plotly::common::Line::new().color("blue").width(1.0))
            .show_legend(false);

        plot.add_trace(trace);
    }

    let layout = Layout::new()
        .title(Title::from(title))
        .x_axis(Axis::new().title("x (μm)"))
        .y_axis(Axis::new().title("y (μm)"))
        .height(600)
        .width(600);

    plot.set_layout(layout);
    plot.write_html(format!("{}.html", filename));
    println!("Saved vector field to {}.html", filename);
}

/// Plot Poynting vector (energy flow)
fn plot_poynting_vector(
    e_field: &WaveArray<Complex64>,
    h_field: &WaveArray<Complex64>,
    slice_index: usize,
    pixel_size: f64,
    title: &str,
    filename: &str,
) {
    let shape = (e_field.shape()[1], e_field.shape()[2], e_field.shape()[3]);

    // Calculate Poynting vector S = Re(E × H*) at z = slice_index
    let mut poynting_x = vec![vec![0.0; shape.1]; shape.0];
    let mut poynting_y = vec![vec![0.0; shape.1]; shape.0];
    let mut poynting_z = vec![vec![0.0; shape.1]; shape.0];

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            let ex = e_field.data[[0, i, j, slice_index]];
            let ey = e_field.data[[1, i, j, slice_index]];
            let ez = e_field.data[[2, i, j, slice_index]];

            let hx = h_field.data[[0, i, j, slice_index]];
            let hy = h_field.data[[1, i, j, slice_index]];
            let hz = h_field.data[[2, i, j, slice_index]];

            // S = E × H* (real part)
            poynting_x[i][j] = (ey * hz.conj() - ez * hy.conj()).re;
            poynting_y[i][j] = (ez * hx.conj() - ex * hz.conj()).re;
            poynting_z[i][j] = (ex * hy.conj() - ey * hx.conj()).re;
        }
    }

    // Plot magnitude of Poynting vector in XY plane
    let mut magnitude = vec![vec![0.0; shape.1]; shape.0];
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            magnitude[i][j] =
                (poynting_x[i][j].powi(2) + poynting_y[i][j].powi(2) + poynting_z[i][j].powi(2))
                    .sqrt();
        }
    }

    let x_axis: Vec<String> = (0..shape.0)
        .map(|i| format!("{:.1}", i as f64 * pixel_size))
        .collect();
    let y_axis: Vec<String> = (0..shape.1)
        .map(|i| format!("{:.1}", i as f64 * pixel_size))
        .collect();

    let trace = HeatMap::new(x_axis, y_axis, magnitude)
        .color_scale(ColorScale::Palette(ColorScalePalette::Turbo))
        .transpose(true)
        .name("Poynting magnitude");

    let layout = Layout::new()
        .title(Title::from(title))
        .x_axis(Axis::new().title("x (μm)"))
        .y_axis(Axis::new().title("y (μm)"))
        .height(600)
        .width(800);

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);

    plot.write_html(format!("{}.html", filename));
    println!("Saved Poynting vector plot to {}.html", filename);
}

fn main() {
    println!("Maxwell 3D Simulation with Visualization");
    println!("========================================\n");

    // Simulation parameters
    let grid_size = 48;
    let shape = (grid_size, grid_size, grid_size);
    let wavelength = 1.55; // μm (telecom wavelength)
    let pixel_size = 0.2; // μm
    let core_width = 3.0; // μm
    let core_height = 2.0; // μm
    let n_core = 1.45; // Silicon dioxide core

    println!("Creating rectangular waveguide structure...");
    println!("  Grid: {}x{}x{}", shape.0, shape.1, shape.2);
    println!("  Wavelength: {} μm", wavelength);
    println!("  Pixel size: {} μm", pixel_size);
    println!("  Core dimensions: {}x{} μm", core_width, core_height);
    println!("  Core refractive index: {}", n_core);

    // Create waveguide medium
    let permittivity = create_waveguide_medium(
        shape,
        (core_width / pixel_size, core_height / pixel_size),
        n_core,
    );

    // Create polarized source (y-polarized for TE-like mode)
    println!("\nCreating y-polarized source...");
    let source = create_polarized_source(
        shape,
        1, // y-component
        (
            core_width / pixel_size * 0.7,
            core_height / pixel_size * 0.7,
        ),
    );

    // Create Maxwell domain
    let domain = MaxwellDomain::new(
        permittivity.clone(),
        pixel_size,
        wavelength,
        [false, false, false],    // Non-periodic
        [[8, 8], [8, 8], [8, 8]], // PML boundaries
    );

    // Solve
    println!("\nSolving Maxwell equations...");
    let config = IterationConfig {
        max_iterations: 500,
        threshold: 1e-5,
        alpha: 0.75,
        full_residuals: false,
    };

    let result = preconditioned_richardson(&domain, &source, config);

    println!("  Converged in {} iterations", result.iterations);
    println!("  Final residual: {:.2e}", result.residual_norm);

    // Calculate H-field from E-field (simplified - normally would use curl operator)
    // H = (1/iωμ₀) ∇ × E
    // For visualization, we'll approximate with a scaled version
    let h_field = result.field.clone(); // Placeholder - would need proper curl calculation

    // Create visualization directory
    std::fs::create_dir_all("plots").unwrap();

    // Plot results
    println!("\nGenerating visualizations...");

    let z_center = shape.2 / 2;
    let y_center = shape.1 / 2;

    // 1. Plot Ex component
    plot_vector_component(
        &result.field,
        0, // Ex
        z_center,
        2, // XY slice
        pixel_size,
        "Ex component - XY plane (z=center)",
        "plots/maxwell_ex",
    );

    // 2. Plot Ey component
    plot_vector_component(
        &result.field,
        1, // Ey
        z_center,
        2, // XY slice
        pixel_size,
        "Ey component - XY plane (z=center)",
        "plots/maxwell_ey",
    );

    // 3. Plot Ez component
    plot_vector_component(
        &result.field,
        2, // Ez
        z_center,
        2, // XY slice
        pixel_size,
        "Ez component - XY plane (z=center)",
        "plots/maxwell_ez",
    );

    // 4. Plot total field intensity
    plot_field_intensity(
        &result.field,
        z_center,
        2,
        pixel_size,
        "Total field intensity |E|² - XY plane",
        "plots/maxwell_intensity",
    );

    // 5. Plot vector field arrows
    plot_vector_field_arrows(
        &result.field,
        z_center,
        pixel_size,
        3, // Step size for arrow spacing
        "Electric field vectors - XY plane",
        "plots/maxwell_vectors",
    );

    // 6. Plot Poynting vector (energy flow)
    plot_poynting_vector(
        &result.field,
        &h_field,
        z_center,
        pixel_size,
        "Poynting vector magnitude - XY plane",
        "plots/maxwell_poynting",
    );

    // 7. Plot waveguide structure
    let mut structure = WaveArray::zeros(shape);
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                structure.data[[i, j, k]] = permittivity.data[[i, j, k]].sqrt();
            }
        }
    }

    plot_vector_component(
        &WaveArray::from_slice(
            &structure.data.as_slice().unwrap(),
            (1, shape.0, shape.1, shape.2),
        ),
        0,
        z_center,
        2,
        pixel_size,
        "Waveguide structure (refractive index)",
        "plots/maxwell_structure",
    );

    // 8. Plot propagation along waveguide (XZ slice)
    plot_vector_component(
        &result.field,
        1, // Ey (main component for TE mode)
        y_center,
        1, // XZ slice
        pixel_size,
        "Ey component - XZ plane (y=center) - Propagation view",
        "plots/maxwell_propagation",
    );

    println!("\nVisualization complete! Check the 'plots' directory for results.");
    println!("Open the .html files in a web browser for interactive plots.");
}
