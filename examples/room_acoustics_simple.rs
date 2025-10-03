//! 2D Room Acoustics Simulation
//!
//! This example simulates acoustic wave propagation in a rectangular room
//! using the 2D Helmholtz equation. It computes the frequency response
//! at a specific point and visualizes SPL and phase vs frequency.

use ndarray::{Array2, Array3};
use num_complex::Complex;
use plotly::{
    common::{ColorScale, ColorScalePalette, Mode, Title},
    layout::{Axis, Layout},
    HeatMap, Plot, Scatter,
};
use std::f64::consts::PI;
use wavesim::{domain::domain_trait::Domain, engine::array::WaveArray};

type Complex64 = Complex<f64>;

/// Room parameters
struct RoomConfig {
    /// Room width in meters
    width: f64,
    /// Room height in meters
    height: f64,
    /// Source position from bottom-left corner (x, y) in meters
    source_pos: (f64, f64),
    /// Measurement point position (x, y) in meters
    measure_pos: (f64, f64),
    /// Wall absorption coefficient (0 = perfect reflection, 1 = full absorption)
    absorption: f64,
    /// Grid points per meter
    grid_resolution: usize,
}

impl Default for RoomConfig {
    fn default() -> Self {
        Self {
            width: 4.0,              // 4 meters
            height: 2.0,             // 2 meters
            source_pos: (0.3, 0.3),  // 30cm from corner
            measure_pos: (2.0, 1.0), // Center of room
            absorption: 0.1,         // 10% absorption
            grid_resolution: 20,     // 20 points per meter (5cm resolution)
        }
    }
}

/// Acoustic source types
#[derive(Clone, Copy)]
enum SourceType {
    Uniform,
    Gaussian(f64), // width parameter
    Dipole,
}

/// Create the room geometry with walls
fn create_room_medium(nx: usize, ny: usize, absorption: f64, freq: f64) -> Array3<Complex64> {
    let mut permittivity = Array3::zeros((nx, ny, 1));

    // Speed of sound in air (m/s)
    let _c_air = 343.0;

    // Convert frequency to angular frequency
    let _omega = 2.0 * PI * freq;

    // Air properties (normalized to c = 1 for numerical stability)
    let n_air = 1.0;

    // Wall properties - using complex refractive index for absorption
    // Higher real part = harder wall (more reflection)
    // Imaginary part = absorption
    let n_wall_real = 10.0; // High impedance mismatch for reflection
    let n_wall_imag = absorption * 2.0; // Absorption coefficient

    // Fill the room
    for i in 0..nx {
        for j in 0..ny {
            // Check if we're at a wall (boundary)
            let is_wall = i == 0 || i == nx - 1 || j == 0 || j == ny - 1;

            if is_wall {
                // Wall with complex refractive index
                let n = Complex::new(n_wall_real, n_wall_imag);
                permittivity[[i, j, 0]] = n * n;
            } else {
                // Air
                permittivity[[i, j, 0]] = Complex::new(n_air * n_air, 0.0);
            }
        }
    }

    permittivity
}

/// Create acoustic source
fn create_acoustic_source(
    nx: usize,
    ny: usize,
    source_pos_grid: (usize, usize),
    source_type: SourceType,
) -> Array3<Complex64> {
    let mut source = Array3::zeros((nx, ny, 1));

    match source_type {
        SourceType::Uniform => {
            // Point source
            source[[source_pos_grid.0, source_pos_grid.1, 0]] = Complex::new(1.0, 0.0);
        }
        SourceType::Gaussian(width) => {
            // Gaussian distributed source
            let (sx, sy) = source_pos_grid;
            let sigma = width;

            for i in 0..nx {
                for j in 0..ny {
                    let dx = i as f64 - sx as f64;
                    let dy = j as f64 - sy as f64;
                    let r2 = dx * dx + dy * dy;
                    let amplitude = (-r2 / (2.0 * sigma * sigma)).exp();

                    if amplitude > 0.01 {
                        // Threshold for computational efficiency
                        source[[i, j, 0]] = Complex::new(amplitude, 0.0);
                    }
                }
            }
        }
        SourceType::Dipole => {
            // Dipole source (two opposite sources)
            let (sx, sy) = source_pos_grid;
            if sx > 0 && sx < nx - 1 {
                source[[sx - 1, sy, 0]] = Complex::new(1.0, 0.0);
                source[[sx + 1, sy, 0]] = Complex::new(-1.0, 0.0);
            }
        }
    }

    source
}

/// Solve Helmholtz equation for a single frequency
fn solve_helmholtz_2d(
    room_config: &RoomConfig,
    frequency: f64,
    source_type: SourceType,
) -> (Array3<Complex64>, usize, f64) {
    // Calculate grid dimensions
    let nx = (room_config.width * room_config.grid_resolution as f64) as usize;
    let ny = (room_config.height * room_config.grid_resolution as f64) as usize;

    // Pixel size in meters
    let pixel_size = 1.0 / room_config.grid_resolution as f64;

    // Wavelength in meters
    let c_sound = 343.0; // Speed of sound in m/s
    let wavelength = c_sound / frequency;

    // Grid positions for source and measurement point
    let source_grid = (
        (room_config.source_pos.0 / pixel_size) as usize,
        (room_config.source_pos.1 / pixel_size) as usize,
    );

    // Create medium
    let permittivity_array = create_room_medium(nx, ny, room_config.absorption, frequency);

    // Create source
    let source_array = create_acoustic_source(nx, ny, source_grid, source_type);

    // Convert to WaveArray (3D arrays with single z-slice for 2D problem)
    let permittivity = WaveArray {
        data: permittivity_array,
    };
    let source = WaveArray { data: source_array };

    // Create Helmholtz domain
    use wavesim::domain::helmholtz::HelmholtzDomain;

    // Add PML boundaries for absorption
    let pml_thickness = 8;
    let domain = HelmholtzDomain::new(
        permittivity,
        pixel_size,
        wavelength,
        [false, false, false], // Non-periodic boundaries (room walls)
        [
            [pml_thickness, pml_thickness],
            [pml_thickness, pml_thickness],
            [0, 0],
        ], // PML on x,y only
    );

    // Solve using preconditioned Richardson iteration
    use wavesim::domain::iteration::{preconditioned_richardson, IterationConfig};

    let config = IterationConfig {
        max_iterations: 1000,
        threshold: 1e-6,
        alpha: 0.75,
        full_residuals: false,
    };

    let result = preconditioned_richardson(&domain, &source, config);

    (result.field.data, result.iterations, result.residual_norm)
}

/// Extract field value at measurement point
fn get_field_at_point(
    field: &Array3<Complex64>,
    room_config: &RoomConfig,
    measure_pos: (f64, f64),
) -> Complex64 {
    let pixel_size = 1.0 / room_config.grid_resolution as f64;
    let i = (measure_pos.0 / pixel_size) as usize;
    let j = (measure_pos.1 / pixel_size) as usize;

    field[[i, j, 0]]
}

/// Calculate SPL (Sound Pressure Level) in dB
fn calculate_spl(pressure: Complex64, reference_pressure: f64) -> f64 {
    let p_rms = pressure.norm() / 2.0_f64.sqrt();
    20.0 * (p_rms / reference_pressure).log10()
}

/// Main frequency sweep analysis
fn frequency_sweep(
    room_config: &RoomConfig,
    frequencies: &[f64],
    source_type: SourceType,
) -> (Vec<f64>, Vec<f64>, Vec<Complex64>) {
    let mut spl_values = Vec::new();
    let mut phase_values = Vec::new();
    let mut complex_values = Vec::new();

    // Reference pressure for SPL (20 μPa for air)
    let p_ref = 20e-6;

    println!("\nFrequency sweep analysis:");
    println!("Frequency (Hz) | SPL (dB) | Phase (deg) | Iterations");
    println!("---------------|----------|-------------|------------");

    for &freq in frequencies {
        let (field, iterations, _residual) = solve_helmholtz_2d(room_config, freq, source_type);

        // Get field at measurement point
        let field_value = get_field_at_point(&field, room_config, room_config.measure_pos);
        complex_values.push(field_value);

        // Calculate SPL
        let spl = calculate_spl(field_value, p_ref);
        spl_values.push(spl);

        // Calculate phase
        let phase_rad = field_value.arg();
        let phase_deg = phase_rad * 180.0 / PI;
        phase_values.push(phase_deg);

        println!(
            "{:14.1} | {:8.2} | {:11.2} | {:10}",
            freq, spl, phase_deg, iterations
        );
    }

    (spl_values, phase_values, complex_values)
}

/// Plot frequency response
fn plot_frequency_response(
    frequencies: &[f64],
    spl_values: &[f64],
    phase_values: &[f64],
    room_config: &RoomConfig,
) {
    // Create SPL plot
    let spl_trace = Scatter::new(frequencies.to_vec(), spl_values.to_vec())
        .mode(Mode::LinesMarkers)
        .name("SPL")
        .line(plotly::common::Line::new().width(2.0));

    let spl_layout = Layout::new()
        .title(Title::from("Sound Pressure Level vs Frequency"))
        .x_axis(
            Axis::new()
                .title("Frequency (Hz)")
                .type_(plotly::layout::AxisType::Log)
                .range(vec![1.3, 3.3]), // log scale from ~20 to ~2000 Hz
        )
        .y_axis(Axis::new().title("SPL (dB)"))
        .height(400);

    let mut spl_plot = Plot::new();
    spl_plot.add_trace(spl_trace.clone());
    spl_plot.set_layout(spl_layout);

    let spl_filename = "plots/room_acoustics_spl.html";
    spl_plot.write_html(spl_filename);
    println!("\nSaved SPL plot to {}", spl_filename);

    // Create Phase plot
    let phase_trace = Scatter::new(frequencies.to_vec(), phase_values.to_vec())
        .mode(Mode::LinesMarkers)
        .name("Phase")
        .line(plotly::common::Line::new().width(2.0).color("red"));

    let phase_layout = Layout::new()
        .title(Title::from("Phase Response vs Frequency"))
        .x_axis(
            Axis::new()
                .title("Frequency (Hz)")
                .type_(plotly::layout::AxisType::Log)
                .range(vec![1.3, 3.3]),
        )
        .y_axis(Axis::new().title("Phase (degrees)"))
        .height(400);

    let mut phase_plot = Plot::new();
    phase_plot.add_trace(phase_trace);
    phase_plot.set_layout(phase_layout);

    let phase_filename = "plots/room_acoustics_phase.html";
    phase_plot.write_html(phase_filename);
    println!("Saved phase plot to {}", phase_filename);

    // Combined plot with dual y-axes
    let combined_layout = Layout::new()
        .title(Title::from(format!(
            "Room Acoustic Response ({}m x {}m, absorption={})",
            room_config.width, room_config.height, room_config.absorption
        )))
        .x_axis(
            Axis::new()
                .title("Frequency (Hz)")
                .type_(plotly::layout::AxisType::Log)
                .range(vec![1.3, 3.3]),
        )
        .y_axis(Axis::new().title("SPL (dB)"))
        .height(500);

    let mut combined_plot = Plot::new();
    combined_plot.add_trace(spl_trace);
    combined_plot.set_layout(combined_layout);

    let combined_filename = "plots/room_acoustics_combined.html";
    combined_plot.write_html(combined_filename);
    println!("Saved combined plot to {}", combined_filename);
}

/// Plot room field distribution at a specific frequency
fn plot_room_field(room_config: &RoomConfig, frequency: f64, source_type: SourceType) {
    let (field, _, _) = solve_helmholtz_2d(room_config, frequency, source_type);

    // Convert to 2D magnitude array
    let nx = (room_config.width * room_config.grid_resolution as f64) as usize;
    let ny = (room_config.height * room_config.grid_resolution as f64) as usize;

    let mut magnitude = vec![];
    for j in 0..ny {
        let mut row = vec![];
        for i in 0..nx {
            row.push(field[[i, j, 0]].norm());
        }
        magnitude.push(row);
    }

    // Create axis labels in meters
    let pixel_size = 1.0 / room_config.grid_resolution as f64;
    let x_labels: Vec<String> = (0..nx)
        .step_by((nx / 20).max(1))
        .map(|i| format!("{:.2}", i as f64 * pixel_size))
        .collect();

    let y_labels: Vec<String> = (0..ny)
        .step_by((ny / 10).max(1))
        .map(|j| format!("{:.2}", j as f64 * pixel_size))
        .collect();

    let trace = HeatMap::new(x_labels, y_labels, magnitude)
        .color_scale(ColorScale::Palette(ColorScalePalette::Viridis));

    let layout = Layout::new()
        .title(Title::from(format!(
            "Room Acoustic Field at {} Hz",
            frequency
        )))
        .x_axis(Axis::new().title("X position (m)"))
        .y_axis(Axis::new().title("Y position (m)"))
        .height(500)
        .width(800);

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);

    let filename = format!("plots/room_field_{:.0}Hz.html", frequency);
    plot.write_html(&filename);
    println!("Saved room field plot to {}", filename);
}

fn main() {
    // Enable parallel computation
    let num_cores = num_cpus::get();
    println!(
        "Using {} threads (all available cores) for parallel computation\n",
        num_cores
    );

    println!("2D Room Acoustics Simulation with Parallel Computation");
    println!("=======================================================");

    // Create room configuration
    let mut room_config = RoomConfig::default();

    println!("\nRoom Configuration:");
    println!(
        "  Dimensions: {} m x {} m",
        room_config.width, room_config.height
    );
    println!(
        "  Source position: ({:.2}, {:.2}) m",
        room_config.source_pos.0, room_config.source_pos.1
    );
    println!(
        "  Measurement point: ({:.2}, {:.2}) m",
        room_config.measure_pos.0, room_config.measure_pos.1
    );
    println!("  Wall absorption: {:.0}%", room_config.absorption * 100.0);
    println!(
        "  Grid resolution: {} points/m ({:.1} cm spacing)",
        room_config.grid_resolution,
        100.0 / room_config.grid_resolution as f64
    );

    // Create plots directory
    std::fs::create_dir_all("plots").unwrap_or(());

    // Define frequency points (logarithmic spacing)
    let frequencies: Vec<f64> = vec![
        20.0, 31.5, 50.0, 80.0, 125.0, 200.0, 315.0, 500.0, 800.0, 1250.0, 2000.0,
    ];

    println!("\nAnalyzing frequencies: {:?} Hz", frequencies);

    // Source type (can be changed)
    let source_type = SourceType::Uniform;

    // Perform frequency sweep
    let (spl_values, phase_values, _complex_values) =
        frequency_sweep(&room_config, &frequencies, source_type);

    // Plot results
    plot_frequency_response(&frequencies, &spl_values, &phase_values, &room_config);

    // Plot field distribution at specific frequencies
    println!("\nGenerating field distributions...");
    for freq in &[125.0, 500.0, 1000.0] {
        plot_room_field(&room_config, *freq, source_type);
    }

    // Calculate room modes (theoretical)
    println!("\nTheoretical room modes:");
    let c = 343.0; // Speed of sound m/s
    for m in 1..=3 {
        for n in 1..=3 {
            let freq = (c / 2.0)
                * ((m as f64 / room_config.width).powi(2)
                    + (n as f64 / room_config.height).powi(2))
                .sqrt();
            if freq <= 2000.0 {
                println!("  Mode ({}, {}): {:.1} Hz", m, n, freq);
            }
        }
    }

    // Try different absorption values
    println!("\nComparing different absorption coefficients...");
    let absorption_values = vec![0.05, 0.2, 0.5];
    let mut spl_comparisons = vec![];

    for &absorption in &absorption_values {
        room_config.absorption = absorption;
        let (spl, _, _) = frequency_sweep(&room_config, &frequencies, source_type);
        spl_comparisons.push((absorption, spl));
    }

    // Plot absorption comparison
    let mut comparison_plot = Plot::new();
    for (absorption, spl) in &spl_comparisons {
        let trace = Scatter::new(frequencies.clone(), spl.clone())
            .mode(Mode::LinesMarkers)
            .name(&format!("α = {:.0}%", absorption * 100.0));
        comparison_plot.add_trace(trace);
    }

    let comparison_layout = Layout::new()
        .title(Title::from("SPL vs Absorption Coefficient"))
        .x_axis(
            Axis::new()
                .title("Frequency (Hz)")
                .type_(plotly::layout::AxisType::Log),
        )
        .y_axis(Axis::new().title("SPL (dB)"))
        .height(500);

    comparison_plot.set_layout(comparison_layout);
    comparison_plot.write_html("plots/room_absorption_comparison.html");

    println!("\nSimulation complete!");
    println!("Check the 'plots' directory for interactive visualizations.");
}
