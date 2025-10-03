//! Analytical Solutions for Helmholtz Equation
//!
//! This example demonstrates the analytical solutions to the Helmholtz equation
//! for simple geometries like rectangles, circles, and spheres.
//!
//! It shows:
//! - How to compute analytical solutions using separation of variables
//! - Comparison with numerical solutions for validation
//! - Visualization of the analytical fields

use num_complex::Complex;
use plotly::{
    common::{ColorScale, ColorScalePalette, Title},
    HeatMap, Layout, Plot, Scatter,
};
use std::f64::consts::PI;
use wavesim::{
    domain::helmholtz::HelmholtzDomain,
    domain::iteration::{preconditioned_richardson, IterationConfig},
    engine::array::{Complex64, WaveArray},
    utilities::analytical::{
        compare_solutions, BoundaryCondition, CircleParams, CircularSolution, RectangleParams,
        RectangularSolution, SphereParams, SphericalSolution,
    },
};

fn main() {
    println!("Analytical Solutions for Helmholtz Equation");
    println!("==========================================\n");

    // Create output directory for plots
    std::fs::create_dir_all("plots/analytical").unwrap();

    // Example 1: Rectangle with Dirichlet boundary conditions
    println!("=== Rectangular Domain ===");
    demonstrate_rectangular_solution();

    // Example 2: Circle with Dirichlet boundary conditions
    println!("\n=== Circular Domain ===");
    demonstrate_circular_solution();

    // Example 3: Sphere with Dirichlet boundary conditions
    println!("\n=== Spherical Domain ===");
    demonstrate_spherical_solution();

    // Example 4: Compare analytical and numerical solutions
    println!("\n=== Numerical vs Analytical Comparison ===");
    compare_analytical_and_numerical();

    println!("\n✓ All analytical solution examples completed!");
    println!("Check the 'plots/analytical' directory for visualizations.");
}

fn demonstrate_rectangular_solution() {
    println!("Creating rectangular analytical solution...");

    // Define rectangular domain: [0, π] × [0, π] × [0, π]
    let params = RectangleParams {
        dimensions: [PI, PI, PI],
        boundary_conditions: [BoundaryCondition::Dirichlet; 6],
        max_modes: [3, 3, 3], // Use first 3 modes in each direction
    };

    let solution = RectangularSolution::new(params);

    // Print eigenvalues
    println!("  First few eigenvalues (k²): ");
    for (i, &eigenvalue) in solution.eigenvalues().iter().take(5).enumerate() {
        println!("    λ_{}: {:.4}", i + 1, eigenvalue);
    }

    // Evaluate solution on a grid
    let grid_size = 32;
    let shape = (grid_size, grid_size, grid_size);
    let grid_spacing = [PI / grid_size as f64; 3];
    let offset = [0.0; 3];

    let field = solution.evaluate_on_grid(shape, grid_spacing, offset);

    // Extract and plot a 2D slice (z = π/2)
    plot_2d_slice(
        &field,
        grid_size / 2,
        PI / grid_size as f64,
        "Rectangular Analytical Solution - XY Slice (z=π/2)",
        "plots/analytical/rectangular_xy_slice",
    );

    // Analyze the field
    let max_amplitude = field.data.iter().map(|c| c.norm()).fold(0.0, f64::max);
    let avg_amplitude = field.data.iter().map(|c| c.norm()).sum::<f64>() / field.data.len() as f64;

    println!("  Max amplitude: {:.4}", max_amplitude);
    println!("  Average amplitude: {:.4}", avg_amplitude);

    // Test specific points
    let center_val = solution.evaluate_at(PI / 2.0, PI / 2.0, PI / 2.0);
    let corner_val = solution.evaluate_at(0.0, 0.0, 0.0);

    println!(
        "  Value at center (π/2, π/2, π/2): {:.4}",
        center_val.norm()
    );
    println!("  Value at corner (0, 0, 0): {:.4}", corner_val.norm());
}

fn demonstrate_circular_solution() {
    println!("Creating circular analytical solution...");

    let params = CircleParams {
        radius: 1.0,
        boundary_condition: BoundaryCondition::Dirichlet,
        max_modes: [3, 5], // [angular_modes, radial_modes]
    };

    let solution = CircularSolution::new(params);

    // Evaluate on a 2D grid
    let grid_size = 64;
    let shape = (grid_size, grid_size);
    let grid_spacing = [2.0 / grid_size as f64; 2];
    let center = [0.0, 0.0];

    let field = solution.evaluate_on_grid_2d(shape, grid_spacing, center);

    // Plot the circular solution
    plot_2d_array(
        &field,
        2.0 / grid_size as f64,
        "Circular Analytical Solution",
        "plots/analytical/circular_solution",
    );

    let max_amplitude = field.iter().map(|c| c.norm()).fold(0.0, f64::max);
    let inside_val = solution.evaluate_at(0.5, 0.0);
    let boundary_val = solution.evaluate_at(1.0, 0.0);

    println!("  Max amplitude: {:.4}", max_amplitude);
    println!("  Value at (0.5, 0): {:.4}", inside_val.norm());
    println!("  Value at boundary (1, 0): {:.4}", boundary_val.norm());
}

fn demonstrate_spherical_solution() {
    println!("Creating spherical analytical solution...");

    let params = SphereParams {
        radius: 1.0,
        boundary_condition: BoundaryCondition::Dirichlet,
        max_modes: [2, 3], // [l_max, radial_modes]
    };

    let solution = SphericalSolution::new(params);

    // Evaluate on a 3D grid
    let grid_size = 32;
    let shape = (grid_size, grid_size, grid_size);
    let grid_spacing = [2.0 / grid_size as f64; 3];
    let center = [0.0, 0.0, 0.0];

    let field = solution.evaluate_on_grid(shape, grid_spacing, center);

    // Plot a slice through the center
    plot_2d_slice(
        &field,
        grid_size / 2,
        2.0 / grid_size as f64,
        "Spherical Analytical Solution - XY Slice (z=0)",
        "plots/analytical/spherical_xy_slice",
    );

    let max_amplitude = field.data.iter().map(|c| c.norm()).fold(0.0, f64::max);
    let center_val = solution.evaluate_at(0.0, 0.0, 0.0);
    let surface_val = solution.evaluate_at(1.0, 0.0, 0.0);

    println!("  Max amplitude: {:.4}", max_amplitude);
    println!("  Value at center (0, 0, 0): {:.4}", center_val.norm());
    println!("  Value at surface (1, 0, 0): {:.4}", surface_val.norm());
}

fn compare_analytical_and_numerical() {
    println!("Comparing analytical and numerical solutions...");

    // Create a simple rectangular domain for comparison
    let domain_size = [PI, PI, PI];
    let grid_size = 32;
    let pixel_size = PI / grid_size as f64;
    let wavelength = 1.0; // This will give us k = 2π

    // Create homogeneous medium (n = 1)
    let shape = (grid_size, grid_size, grid_size);
    let permittivity = WaveArray::from_scalar(shape, Complex64::new(1.0, 0.0));

    // Create a source that matches our analytical solution structure
    // For simplicity, use a point source at the center
    let mut source = WaveArray::zeros(shape);
    source.data[[grid_size / 2, grid_size / 2, grid_size / 2]] = Complex64::new(1.0, 0.0);

    // Set up numerical solver
    let domain = HelmholtzDomain::new(
        permittivity,
        pixel_size,
        wavelength,
        [false, false, false],    // Non-periodic boundaries
        [[4, 4], [4, 4], [4, 4]], // PML boundaries
    );

    println!("  Running numerical solver...");
    let config = IterationConfig {
        max_iterations: 200,
        threshold: 1e-6,
        alpha: 0.75,
        full_residuals: false,
    };

    let result = preconditioned_richardson(&domain, &source, config);
    println!("    Converged in {} iterations", result.iterations);
    println!("    Final residual: {:.2e}", result.residual_norm);

    // Create analytical solution
    println!("  Computing analytical solution...");
    let rect_params = RectangleParams {
        dimensions: domain_size,
        boundary_conditions: [BoundaryCondition::Dirichlet; 6],
        max_modes: [2, 2, 2], // Use fewer modes for better comparison
    };

    let analytical_solution = RectangularSolution::new(rect_params);
    let grid_spacing = [pixel_size; 3];
    let offset = [0.0; 3];

    let analytical_field = analytical_solution.evaluate_on_grid(shape, grid_spacing, offset);

    // Compare solutions
    let (l2_error, max_error, rel_error) = compare_solutions(&result.field, &analytical_field);

    println!("  Comparison results:");
    println!("    L2 error: {:.2e}", l2_error);
    println!("    Max error: {:.2e}", max_error);
    println!("    Relative L2 error: {:.2e}", rel_error);

    // Plot both solutions for visual comparison
    plot_2d_slice(
        &result.field,
        grid_size / 2,
        pixel_size,
        "Numerical Solution - XY Slice (z=π/2)",
        "plots/analytical/numerical_solution",
    );

    plot_2d_slice(
        &analytical_field,
        grid_size / 2,
        pixel_size,
        "Analytical Solution - XY Slice (z=π/2)",
        "plots/analytical/analytical_solution",
    );

    // Plot the error field
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
        grid_size / 2,
        pixel_size,
        "Error Field (Numerical - Analytical)",
        "plots/analytical/error_field",
    );
}

/// Plot a 2D slice from a 3D field
fn plot_2d_slice(
    field: &WaveArray<Complex64>,
    slice_z: usize,
    pixel_size: f64,
    title: &str,
    filename: &str,
) {
    let shape = field.shape_tuple();
    let (nx, ny, _nz) = shape;

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
        .step_by((nx / 10).max(1))
        .map(|i| format!("{:.2}", i as f64 * pixel_size))
        .collect();

    let y_labels: Vec<String> = (0..ny)
        .step_by((ny / 10).max(1))
        .map(|j| format!("{:.2}", j as f64 * pixel_size))
        .collect();

    let trace = HeatMap::new(x_labels, y_labels, magnitude)
        .color_scale(ColorScale::Palette(ColorScalePalette::Viridis));

    let layout = Layout::new()
        .title(Title::from(title))
        .x_axis(plotly::layout::Axis::new().title("x"))
        .y_axis(plotly::layout::Axis::new().title("y"))
        .width(600)
        .height(600);

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);

    let html_path = format!("{}.html", filename);
    plot.write_html(&html_path);
    println!("  Saved plot: {}", html_path);
}

/// Plot a 2D array
fn plot_2d_array(field: &ndarray::Array2<Complex64>, pixel_size: f64, title: &str, filename: &str) {
    let shape = field.dim();

    let mut magnitude = vec![];
    for i in 0..shape.0 {
        let mut row = vec![];
        for j in 0..shape.1 {
            let value = field[[i, j]].norm();
            row.push(value);
        }
        magnitude.push(row);
    }

    let x_labels: Vec<String> = (0..shape.0)
        .step_by((shape.0 / 10).max(1))
        .map(|i| format!("{:.2}", (i as f64 - shape.0 as f64 / 2.0) * pixel_size))
        .collect();

    let y_labels: Vec<String> = (0..shape.1)
        .step_by((shape.1 / 10).max(1))
        .map(|j| format!("{:.2}", (j as f64 - shape.1 as f64 / 2.0) * pixel_size))
        .collect();

    let trace = HeatMap::new(x_labels, y_labels, magnitude)
        .color_scale(ColorScale::Palette(ColorScalePalette::Viridis));

    let layout = Layout::new()
        .title(Title::from(title))
        .x_axis(plotly::layout::Axis::new().title("x"))
        .y_axis(plotly::layout::Axis::new().title("y"))
        .width(600)
        .height(600);

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);

    let html_path = format!("{}.html", filename);
    plot.write_html(&html_path);
    println!("  Saved plot: {}", html_path);
}
