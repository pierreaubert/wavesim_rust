//! Room Acoustics Simulation (2D or 3D)
//!
//! Configurable room acoustics simulation with support for 2D and 3D modes

use clap::Parser;
use ndarray::Array3;
use num_complex::Complex;
use plotly::{
    common::{ColorScale, ColorScalePalette, Mode, Title},
    layout::{Axis, Layout, Legend, Margin},
    HeatMap, Plot, Scatter,
};
use std::env;
use std::f64::consts::PI;
use wavesim::{
    domain::block_decomposition::solve_helmholtz_block,
    domain::helmholtz_schwarz::solve_helmholtz_schwarz,
    engine::array::WaveArray,
    utilities::analytical::{BoundaryCondition, RectangleParams, RectangularSolution},
};

type Complex64 = Complex<f64>;

const REFERENCEPRESSURE: f64 = 20e-6; // Reference pressure in Pa

/// Command-line arguments for room acoustics simulation
#[derive(Parser, Debug)]
#[command(name = "room_acoustics")]
#[command(about = "Room Acoustics Simulation with Helmholtz Equation (2D or 3D)", long_about = None)]
struct Args {
    /// Simulation mode: 2d or 3d
    #[arg(long, default_value = "2d")]
    mode: String,
    /// Minimum frequency in Hz
    #[arg(long, default_value_t = 50.0)]
    min_freq: f64,

    /// Maximum frequency in Hz
    #[arg(long, default_value_t = 2000.0)]
    max_freq: f64,

    /// Number of frequency points
    #[arg(long, default_value_t = 6)]
    nb_freq: usize,

    /// Number of sources (1 or 2)
    #[arg(long, default_value_t = 1)]
    num_sources: usize,

    /// Source 1 position X in meters
    #[arg(long, default_value_t = 0.3)]
    source1_x: f64,

    /// Source 1 position Y in meters
    #[arg(long, default_value_t = 0.3)]
    source1_y: f64,

    /// Source 1 position Z in meters (3D only)
    #[arg(long, default_value_t = 1.0)]
    source1_z: f64,

    /// Source 2 position X in meters (if num_sources=2)
    #[arg(long, default_value_t = 3.7)]
    source2_x: f64,

    /// Source 2 position Y in meters (if num_sources=2)
    #[arg(long, default_value_t = 1.7)]
    source2_y: f64,

    /// Source 2 position Z in meters (if num_sources=2, 3D only)
    #[arg(long, default_value_t = 1.0)]
    source2_z: f64,

    /// Measurement point position X in meters
    #[arg(long, default_value_t = 2.0)]
    measure_x: f64,

    /// Measurement point position Y in meters
    #[arg(long, default_value_t = 1.0)]
    measure_y: f64,

    /// Measurement point position Z in meters (3D only)
    #[arg(long, default_value_t = 1.5)]
    measure_z: f64,

    /// Room width in meters (X dimension)
    #[arg(long, default_value_t = 4.0)]
    room_width: f64,

    /// Room depth in meters (Y dimension)
    #[arg(long, default_value_t = 2.0)]
    room_depth: f64,

    /// Room height in meters (Z dimension, 3D only)
    #[arg(long, default_value_t = 3.0)]
    room_height: f64,

    /// Wall absorption coefficient (0.0 to 1.0)
    #[arg(long, default_value_t = 0.1)]
    absorption: f64,

    /// Grid resolution (points per meter)
    #[arg(long, default_value_t = 10)]
    grid_resolution: usize,

    /// Number of subdomains for domain decomposition (0 = auto, based on grid size)
    #[arg(long, default_value_t = 0)]
    num_subdomains: usize,

    /// Room shape: rectangle, circle (2D), or sphere (3D)
    #[arg(long, default_value = "rectangle")]
    shape: String,

    /// Source sound pressure level in dB SPL (at source location)
    #[arg(long, default_value_t = 94.0)]
    source_spl: f64,

    /// Domain decomposition method: block (default) or schwarz
    #[arg(long, default_value = "block")]
    decomposition_method: String,

    /// Compare with analytical solutions (rectangular rooms only)
    #[arg(long, default_value_t = false)]
    compare_analytical: bool,
}

/// Room configuration
struct RoomConfig {
    mode: SimulationMode,
    shape: RoomShape,
    width: f64,                             // meters (X)
    depth: f64,                             // meters (Y)
    height: f64,                            // meters (Z, 3D only)
    source_positions: Vec<(f64, f64, f64)>, // (x, y, z)
    measure_pos: (f64, f64, f64),           // (x, y, z)
    absorption: f64,
    max_grid_resolution: usize, // maximum points per meter (command-line)
    source_spl: f64,
    decomposition_method: String, // Domain decomposition method
    compare_analytical: bool,     // Whether to compare with analytical solutions
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SimulationMode {
    TwoD,
    ThreeD,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum RoomShape {
    Rectangle,
    Circle, // 2D circular room
    Sphere, // 3D spherical room
}

impl RoomConfig {
    fn from_args(args: &Args) -> Self {
        let mode = match args.mode.to_lowercase().as_str() {
            "2d" => SimulationMode::TwoD,
            "3d" => SimulationMode::ThreeD,
            _ => {
                eprintln!("Invalid mode '{}', using 2D", args.mode);
                SimulationMode::TwoD
            }
        };

        let shape = match args.shape.to_lowercase().as_str() {
            "rectangle" => RoomShape::Rectangle,
            "circle" => {
                if mode == SimulationMode::ThreeD {
                    eprintln!("Warning: Circle shape only valid for 2D mode, using Rectangle");
                    RoomShape::Rectangle
                } else {
                    RoomShape::Circle
                }
            }
            "sphere" => {
                if mode == SimulationMode::TwoD {
                    eprintln!("Warning: Sphere shape only valid for 3D mode, using Rectangle");
                    RoomShape::Rectangle
                } else {
                    RoomShape::Sphere
                }
            }
            _ => {
                eprintln!("Invalid shape '{}', using Rectangle", args.shape);
                RoomShape::Rectangle
            }
        };

        let mut source_positions = vec![(args.source1_x, args.source1_y, args.source1_z)];

        if args.num_sources == 2 {
            source_positions.push((args.source2_x, args.source2_y, args.source2_z));
        }

        Self {
            mode,
            shape,
            width: args.room_width,
            depth: args.room_depth,
            height: args.room_height,
            source_positions,
            measure_pos: (args.measure_x, args.measure_y, args.measure_z),
            absorption: args.absorption,
            max_grid_resolution: args.grid_resolution,
            source_spl: args.source_spl,
            decomposition_method: args.decomposition_method.clone(),
            compare_analytical: args.compare_analytical,
        }
    }
}

/// Calculate optimal grid resolution for a given frequency
fn calculate_optimal_resolution(freq: f64, max_resolution: usize, target_ppw: f64) -> usize {
    let c_sound = 343.0; // m/s
    let wavelength = c_sound / freq;

    // Calculate minimum resolution needed for target points per wavelength
    let min_resolution = (target_ppw / wavelength).ceil() as usize;

    // Use the minimum of max_resolution and what's needed
    min_resolution.min(max_resolution).max(16) // At least N points/m
}

/// Create room medium
fn create_room(
    nx: usize,
    ny: usize,
    nz: usize,
    mode: SimulationMode,
    room_shape: RoomShape,
    absorption: f64,
) -> WaveArray<Complex64> {
    let shape = (nx, ny, nz);
    let mut data = Array3::zeros(shape);

    let n_air = 1.0;
    let n_wall = Complex::new(5.0, absorption); // Wall properties

    match (mode, room_shape) {
        (SimulationMode::TwoD, RoomShape::Rectangle) => {
            // 2D rectangular room: walls at boundaries
            for i in 0..nx {
                for j in 0..ny {
                    let is_wall = i < 2 || i >= nx - 2 || j < 2 || j >= ny - 2;

                    if is_wall {
                        data[[i, j, 0]] = n_wall * n_wall;
                    } else {
                        data[[i, j, 0]] = Complex::new(n_air * n_air, 0.0);
                    }
                }
            }
        }
        (SimulationMode::TwoD, RoomShape::Circle) => {
            // 2D circular room: wall is a circle
            let center_x = nx as f64 / 2.0;
            let center_y = ny as f64 / 2.0;
            let radius = (nx.min(ny) as f64 / 2.0) - 2.0; // 2 pixel wall thickness

            for i in 0..nx {
                for j in 0..ny {
                    let dx = i as f64 + 0.5 - center_x;
                    let dy = j as f64 + 0.5 - center_y;
                    let dist = (dx * dx + dy * dy).sqrt();

                    if dist > radius {
                        data[[i, j, 0]] = n_wall * n_wall;
                    } else {
                        data[[i, j, 0]] = Complex::new(n_air * n_air, 0.0);
                    }
                }
            }
        }
        (SimulationMode::ThreeD, RoomShape::Rectangle) => {
            // 3D rectangular room: walls at boundaries
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let is_wall =
                            i < 2 || i >= nx - 2 || j < 2 || j >= ny - 2 || k < 2 || k >= nz - 2;

                        if is_wall {
                            data[[i, j, k]] = n_wall * n_wall;
                        } else {
                            data[[i, j, k]] = Complex::new(n_air * n_air, 0.0);
                        }
                    }
                }
            }
        }
        (SimulationMode::ThreeD, RoomShape::Sphere) => {
            // 3D spherical room: wall is a sphere
            let center_x = nx as f64 / 2.0;
            let center_y = ny as f64 / 2.0;
            let center_z = nz as f64 / 2.0;
            let radius = (nx.min(ny).min(nz) as f64 / 2.0) - 2.0; // 2 pixel wall thickness

            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let dx = i as f64 + 0.5 - center_x;
                        let dy = j as f64 + 0.5 - center_y;
                        let dz = k as f64 + 0.5 - center_z;
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                        if dist > radius {
                            data[[i, j, k]] = n_wall * n_wall;
                        } else {
                            data[[i, j, k]] = Complex::new(n_air * n_air, 0.0);
                        }
                    }
                }
            }
        }
        _ => {
            // Invalid combination, default to rectangular
            eprintln!("Warning: Invalid mode/shape combination, using rectangular");
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        data[[i, j, k]] = Complex::new(n_air * n_air, 0.0);
                    }
                }
            }
        }
    }

    WaveArray { data }
}

/// Create source(s) - can handle single or multiple sources
/// source_amplitude is the pressure amplitude in Pascals
fn create_sources(
    nx: usize,
    ny: usize,
    nz: usize,
    positions: &[(usize, usize, usize)],
    source_amplitude: f64,
) -> WaveArray<Complex64> {
    let shape = (nx, ny, nz);
    let mut data = Array3::zeros(shape);

    for &pos in positions {
        data[[pos.0, pos.1, pos.2]] = Complex::new(source_amplitude, 0.0);
    }

    WaveArray { data }
}

/// Calculate optimal number of subdomains based on grid size
fn calculate_subdomains(
    nx: usize,
    ny: usize,
    nz: usize,
    mode: SimulationMode,
    num_subdomains: usize,
) -> (usize, usize, usize) {
    if num_subdomains > 0 {
        // User specified, use cubic root to distribute
        let sub_per_dim = (num_subdomains as f64).powf(1.0 / 3.0).ceil() as usize;
        return match mode {
            SimulationMode::TwoD => (sub_per_dim, sub_per_dim, 1),
            SimulationMode::ThreeD => (sub_per_dim, sub_per_dim, sub_per_dim),
        };
    }

    // Auto-calculate based on grid size
    // Use domain decomposition if any dimension > 40 cells
    let total_cells = nx * ny * nz;

    if total_cells < 1000 {
        // Small grid, no decomposition
        (1, 1, 1)
    } else if total_cells < 10000 {
        // Medium grid, 2x2 or 2x2x2
        match mode {
            SimulationMode::TwoD => (2, 2, 1),
            SimulationMode::ThreeD => (2, 2, 2),
        }
    } else {
        // Large grid, more subdomains
        match mode {
            SimulationMode::TwoD => (4, 4, 1),
            SimulationMode::ThreeD => (3, 3, 3),
        }
    }
}

/// Solve for one frequency and return full field with metadata
fn solve_frequency_full_field(
    config: &RoomConfig,
    freq: f64,
    num_subdomains_arg: usize,
) -> (
    WaveArray<Complex64>,
    usize,
    usize,
    (usize, usize, usize),
    f64,
) {
    use wavesim::domain::helmholtz::HelmholtzDomain;
    use wavesim::domain::iteration::{preconditioned_richardson, IterationConfig};

    // Calculate optimal grid resolution for this frequency (target 8 points per wavelength)
    let grid_resolution = calculate_optimal_resolution(freq, config.max_grid_resolution, 8.0);

    let nx = (config.width * grid_resolution as f64) as usize;
    let ny = (config.depth * grid_resolution as f64) as usize;
    let nz = match config.mode {
        SimulationMode::TwoD => 1,
        SimulationMode::ThreeD => (config.height * grid_resolution as f64) as usize,
    };
    let pixel_size = 1.0 / grid_resolution as f64;

    let c_sound = 343.0; // m/s
    let wavelength = c_sound / freq;

    // Validate mesh resolution: need at least 4 points per wavelength (ideally more)
    let points_per_wavelength = wavelength / pixel_size;
    if points_per_wavelength < 4.0 {
        eprintln!(
            "\n⚠️  WARNING: At {:.1} Hz, wavelength is {:.3} m but pixel size is {:.3} m",
            freq, wavelength, pixel_size
        );
        eprintln!(
            "   Only {:.1} points per wavelength (need at least 4, ideally 8+)",
            points_per_wavelength
        );
        eprintln!(
            "   Solution may be inaccurate. Increase --grid-resolution or reduce frequency.\n"
        );
    }

    // Create medium and source(s)
    let permittivity = create_room(nx, ny, nz, config.mode, config.shape, config.absorption);

    // Convert all source positions to grid coordinates
    let source_grids: Vec<(usize, usize, usize)> = config
        .source_positions
        .iter()
        .map(|&(x, y, z)| {
            (
                (x / pixel_size) as usize,
                (y / pixel_size) as usize,
                if config.mode == SimulationMode::TwoD {
                    0
                } else {
                    (z / pixel_size) as usize
                },
            )
        })
        .collect();

    // Convert source SPL to pressure amplitude
    let source_amplitude = spl_to_pressure(config.source_spl);
    let source = create_sources(nx, ny, nz, &source_grids, source_amplitude);

    // Calculate optimal subdomain decomposition
    let subdomains = calculate_subdomains(nx, ny, nz, config.mode, num_subdomains_arg);
    let use_decomposition = subdomains.0 * subdomains.1 * subdomains.2 > 1;

    // Create domain with appropriate boundaries
    let boundaries = match config.mode {
        SimulationMode::TwoD => [[4, 4], [4, 4], [0, 0]],
        SimulationMode::ThreeD => [[4, 4], [4, 4], [4, 4]],
    };

    let domain = HelmholtzDomain::new(
        permittivity,
        pixel_size,
        wavelength,
        [false, false, false],
        boundaries,
    );

    // Use domain decomposition for large grids, standard solver for small grids
    let result = if use_decomposition {
        // Use Schwarz domain decomposition for parallel solving
        // For 4x4 decomposition, increase iterations and relax tolerance
        let max_iters = if subdomains.0 >= 4 || subdomains.1 >= 4 {
            200
        } else {
            100
        };
        let tolerance = if subdomains.0 >= 4 || subdomains.1 >= 4 {
            1e-4
        } else {
            1e-5
        };

        let solution = if config.decomposition_method == "schwarz" {
            println!("Using Schwarz domain decomposition (WARNING: known issues with this method)");
            solve_helmholtz_schwarz(domain, source.clone(), subdomains, max_iters, tolerance)
        } else {
            println!("Using block domain decomposition");
            solve_helmholtz_block(domain, source.clone(), subdomains, max_iters, tolerance)
        };

        // Create a pseudo-result struct
        wavesim::domain::iteration::IterationResult {
            field: solution,
            iterations: max_iters,
            residual_norm: 0.0,
            residual_history: None,
        }
    } else {
        // Small grid: use standard solver
        let iter_config = IterationConfig {
            max_iterations: 1000,
            threshold: 1e-5,
            alpha: 0.75,
            full_residuals: false,
        };

        preconditioned_richardson(&domain, &source, iter_config)
    };

    // Return full field with metadata
    (
        result.field,
        result.iterations,
        grid_resolution,
        subdomains,
        pixel_size,
    )
}

/// Solve for one frequency and return field at measurement point only
fn _solve_frequency(
    config: &RoomConfig,
    freq: f64,
    num_subdomains_arg: usize,
) -> (Complex64, usize, usize, (usize, usize, usize)) {
    let (field, iterations, grid_resolution, subdomains, pixel_size) =
        solve_frequency_full_field(config, freq, num_subdomains_arg);

    // Extract field at measurement point
    let measure_grid = (
        (config.measure_pos.0 / pixel_size) as usize,
        (config.measure_pos.1 / pixel_size) as usize,
        if config.mode == SimulationMode::TwoD {
            0
        } else {
            (config.measure_pos.2 / pixel_size) as usize
        },
    );

    let field_value = field.data[[measure_grid.0, measure_grid.1, measure_grid.2]];

    (field_value, iterations, grid_resolution, subdomains)
}

/// Convert SPL (dB) to pressure amplitude (Pa)
fn spl_to_pressure(spl_db: f64) -> f64 {
    REFERENCEPRESSURE * 10f64.powf(spl_db / 20.0)
}

/// Calculate SPL from pressure amplitude
fn pressure_to_spl(pressure: Complex64) -> f64 {
    let p_rms = pressure.norm();
    20.0 * (p_rms / REFERENCEPRESSURE).log10()
}

/// Plot a 2D field slice
fn _plot_field_2d(
    field: &WaveArray<Complex64>,
    pixel_size: f64,
    z_slice: usize,
    freq: f64,
    mode: SimulationMode,
    room_shape: RoomShape,
    room_width: f64,
    room_depth: f64,
    filename: &str,
) {
    let (nx, ny, _nz) = (
        field.data.shape()[0],
        field.data.shape()[1],
        field.data.shape()[2],
    );

    // Convert to magnitude grid for plotting
    let mut magnitude = vec![];
    for j in 0..ny {
        let mut row = vec![];
        for i in 0..nx {
            let val = field.data[[i, j, z_slice]].norm();
            // Mask out regions outside circle/sphere
            let masked_val = if room_shape == RoomShape::Circle {
                // Calculate position in physical space (meters)
                let x_meters = (i as f64 / nx as f64) * room_width;
                let y_meters = (j as f64 / ny as f64) * room_depth;

                // Circle center and radius in physical space
                let center_x_m = room_width / 2.0;
                let center_y_m = room_depth / 2.0;
                let radius_m = (room_width.min(room_depth) / 2.0) - (2.0 * pixel_size);

                let dx = x_meters - center_x_m;
                let dy = y_meters - center_y_m;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist > radius_m {
                    f64::NAN // Mask outside circle
                } else {
                    val
                }
            } else {
                val
            };

            // Convert to dB SPL
            if masked_val.is_nan() {
                row.push(f64::NAN);
            } else {
                row.push(20.0 * (masked_val / REFERENCEPRESSURE).log10());
            }
        }
        magnitude.push(row);
    }

    // Create numeric x/y coordinates in meters
    let x_coords: Vec<f64> = (0..nx)
        .map(|i| (i as f64 / nx as f64) * room_width)
        .collect();

    let y_coords: Vec<f64> = (0..ny)
        .map(|j| (j as f64 / ny as f64) * room_depth)
        .collect();

    let trace = HeatMap::new_z(magnitude)
        .x(x_coords)
        .y(y_coords)
        .color_scale(ColorScale::Palette(ColorScalePalette::Jet));

    let shape_str = match room_shape {
        RoomShape::Rectangle => "Rectangular",
        RoomShape::Circle => "Circular",
        RoomShape::Sphere => "Spherical",
    };

    let title = if mode == SimulationMode::ThreeD {
        format!(
            "{} Room Field at {:.0} Hz (dB SPL) - Z slice at {:.2}m",
            shape_str,
            freq,
            z_slice as f64 * pixel_size
        )
    } else {
        format!("{} Room Field at {:.0} Hz (dB SPL)", shape_str, freq)
    };

    // Calculate plot size to maintain aspect ratio
    let aspect_ratio = room_width / room_depth;
    let plot_height = 600;
    let plot_width = (plot_height as f64 * aspect_ratio) as usize;

    let layout = Layout::new()
        .title(Title::from(title))
        .x_axis(Axis::new().title("X (m)").range(vec![0.0, room_width]))
        .y_axis(Axis::new().title("Y (m)").range(vec![0.0, room_depth]))
        .height(plot_height)
        .width(plot_width.max(400).min(1000));

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);

    plot.write_html(filename);
}

/// Plot field at specific frequency (convenience function)
fn _plot_field_at_frequency(config: &RoomConfig, freq: f64, filename: &str) {
    // Solve for full field
    let (field, _iterations, _grid_resolution, _subdomains, pixel_size) =
        solve_frequency_full_field(config, freq, 0);

    let nz = field.data.shape()[2];

    // For 3D, take a slice at the middle Z plane
    let z_slice = if config.mode == SimulationMode::TwoD {
        0
    } else {
        nz / 2
    };

    // Plot the field
    _plot_field_2d(
        &field,
        pixel_size,
        z_slice,
        freq,
        config.mode,
        config.shape,
        config.width,
        config.depth,
        filename,
    );
}

/// Plot multiple frequency fields in a single HTML file with grid layout
fn plot_fields_multi_frequency(
    fields: &[(WaveArray<Complex64>, f64, f64, usize)], // (field, freq, pixel_size, z_slice)
    mode: SimulationMode,
    room_shape: RoomShape,
    room_width: f64,
    room_depth: f64,
    filename: &str,
) {
    use plotly::layout::{GridPattern, LayoutGrid};

    let n_fields = fields.len();
    if n_fields == 0 {
        return;
    }

    // Layout: one plot per row
    let cols = 1;
    let rows = n_fields;

    let mut plot = Plot::new();

    // Create a trace for each field
    for (idx, (field, _freq, pixel_size, z_slice)) in fields.iter().enumerate() {
        let (nx, ny, _nz) = (
            field.data.shape()[0],
            field.data.shape()[1],
            field.data.shape()[2],
        );

        // Convert to magnitude grid
        let mut magnitude = vec![];
        for j in 0..ny {
            let mut row = vec![];
            for i in 0..nx {
                let val = field.data[[i, j, *z_slice]].norm();

                // Mask out regions outside circle/sphere
                let masked_val = if room_shape == RoomShape::Circle {
                    // Calculate position in physical space (meters)
                    let x_meters = (i as f64 / nx as f64) * room_width;
                    let y_meters = (j as f64 / ny as f64) * room_depth;

                    // Circle center and radius in physical space
                    let center_x_m = room_width / 2.0;
                    let center_y_m = room_depth / 2.0;
                    let radius_m = (room_width.min(room_depth) / 2.0) - (2.0 * pixel_size);

                    let dx = x_meters - center_x_m;
                    let dy = y_meters - center_y_m;
                    let dist = (dx * dx + dy * dy).sqrt();

                    if dist > radius_m {
                        f64::NAN
                    } else {
                        val
                    }
                } else {
                    val
                };

                // Convert to dB SPL
                if masked_val.is_nan() {
                    row.push(f64::NAN);
                } else {
                    row.push(20.0 * masked_val.log10()); // Relative dB
                }
            }
            magnitude.push(row);
        }

        // Create numeric x/y coordinates in meters
        let x_coords: Vec<f64> = (0..nx)
            .map(|i| (i as f64 / nx as f64) * room_width)
            .collect();

        let y_coords: Vec<f64> = (0..ny)
            .map(|j| (j as f64 / ny as f64) * room_depth)
            .collect();

        // Create heatmap trace
        let trace = HeatMap::new_z(magnitude)
            .x(x_coords)
            .y(y_coords)
            .color_scale(ColorScale::Palette(ColorScalePalette::Jet))
            .x_axis(format!("x{}", idx + 1))
            .y_axis(format!("y{}", idx + 1))
            .show_scale(false); // Hide individual colorbars to save space

        plot.add_trace(trace);
    }

    // Create grid layout - one plot per row, preserve aspect ratio
    // Calculate plot size to maintain aspect ratio based on room dimensions
    let aspect_ratio = room_width / room_depth;
    let plot_height = 500;
    let plot_width = (plot_height as f64 * aspect_ratio) as usize;

    let layout = Layout::new()
        .title(Title::from(format!(
            "Room Acoustic Fields - {} Frequencies",
            n_fields
        )))
        .height(plot_height * rows + 100) // 500px per plot + margin
        .width(plot_width.max(400).min(800)) // Constrain between 400-800px
        .grid(
            LayoutGrid::new()
                .rows(rows)
                .columns(cols)
                .pattern(GridPattern::Independent),
        );

    // Set individual subplot titles and axes
    let mut layout = layout;
    for (idx, (_field, freq, _pixel_size, z_slice)) in fields.iter().enumerate() {
        let row = idx / cols;
        let col = idx % cols;

        let title = if mode == SimulationMode::ThreeD {
            format!("{:.0} Hz (z={:.2}m)", freq, *z_slice as f64 * _pixel_size)
        } else {
            format!("{:.0} Hz", freq)
        };

        // Add annotation for each subplot title
        let annotation = plotly::layout::Annotation::new()
            .x_ref("paper")
            .y_ref("paper")
            .x((col as f64 + 0.5) / cols as f64)
            .y(1.0 - (row as f64) / rows as f64)
            .text(title)
            .font(plotly::common::Font::new().size(12));
        layout.add_annotation(annotation);
    }

    plot.set_layout(layout);
    plot.write_html(filename);
}

/// Analytical solution wrapper for room acoustics
mod analytical_room {
    use super::*;

    #[derive(Debug)]
    pub struct AnalyticalComparison {
        pub l2_error: f64,
        pub max_error: f64,
        pub relative_l2_error: f64,
        pub analytical_field: WaveArray<Complex64>,
    }

    /// Compute analytical solution for rectangular room
    pub fn compute_analytical_solution(
        config: &RoomConfig,
        freq: f64,
        pixel_size: f64,
        grid_shape: (usize, usize, usize),
    ) -> Option<WaveArray<Complex64>> {
        // Only support rectangular rooms
        if config.shape != RoomShape::Rectangle {
            return None;
        }

        // Convert frequency to wavenumber
        let c_sound = 343.0; // m/s
        let wavelength = c_sound / freq;
        let k = 2.0 * PI / wavelength;

        // Create analytical solution parameters for rectangular room
        // Use Neumann boundary conditions (rigid walls) typical for room acoustics
        let params = RectangleParams {
            dimensions: [config.width, config.depth, config.height],
            boundary_conditions: [BoundaryCondition::Neumann; 6],
            max_modes: [
                5,
                5,
                if config.mode == SimulationMode::TwoD {
                    1
                } else {
                    5
                },
            ],
        };

        let mut solution = RectangularSolution::new(params);

        // For room acoustics, we need to set appropriate coefficients
        // This is a simplified approach - in practice, coefficients would be
        // determined by the specific source and boundary conditions
        let num_modes = solution.eigenvalues().len();
        let mut coefficients = vec![Complex64::new(0.0, 0.0); num_modes];

        // Set a few low-frequency modes with simple coefficients
        // This is a heuristic approach for demonstration
        for (i, coeff) in coefficients.iter_mut().enumerate().take(8.min(num_modes)) {
            let decay_factor = (-0.5 * i as f64).exp(); // Exponential decay
            *coeff = Complex64::new(decay_factor, 0.0);
        }

        solution.set_coefficients(coefficients);

        // Evaluate solution on grid
        let grid_spacing = [pixel_size; 3];
        let offset = [0.0; 3];

        Some(solution.evaluate_on_grid(grid_shape, grid_spacing, offset))
    }

    /// Compare numerical and analytical solutions
    pub fn compare_solutions(
        numerical: &WaveArray<Complex64>,
        analytical: &WaveArray<Complex64>,
    ) -> AnalyticalComparison {
        let (l2_error, max_error, relative_l2_error) =
            wavesim::utilities::analytical::compare_solutions(numerical, analytical);

        AnalyticalComparison {
            l2_error,
            max_error,
            relative_l2_error,
            analytical_field: analytical.clone(),
        }
    }

    /// Plot comparison between numerical and analytical solutions
    pub fn plot_comparison(
        numerical: &WaveArray<Complex64>,
        analytical: &WaveArray<Complex64>,
        error_field: &WaveArray<Complex64>,
        pixel_size: f64,
        freq: f64,
        slice_z: usize,
        config: &RoomConfig,
    ) {
        let shape = numerical.shape_tuple();
        let (nx, ny, _nz) = shape;

        // Create three plots side by side
        let plots_data = vec![
            ("Numerical", numerical),
            ("Analytical", analytical),
            ("Error (Num-Ana)", error_field),
        ];

        for (i, (name, field)) in plots_data.iter().enumerate() {
            let mut magnitude = vec![];

            for j in 0..ny {
                let mut row = vec![];
                for k in 0..nx {
                    let val = field.data[[k, j, slice_z]].norm();
                    // Convert to dB SPL for numerical/analytical, linear scale for error
                    let db_val = if i < 2 {
                        20.0 * (val / REFERENCEPRESSURE).log10()
                    } else {
                        val // Linear scale for error
                    };
                    row.push(db_val);
                }
                magnitude.push(row);
            }

            // Create coordinate arrays
            let x_coords: Vec<f64> = (0..nx)
                .map(|k| (k as f64 / nx as f64) * config.width)
                .collect();
            let y_coords: Vec<f64> = (0..ny)
                .map(|j| (j as f64 / ny as f64) * config.depth)
                .collect();

            let trace = HeatMap::new_z(magnitude)
                .x(x_coords)
                .y(y_coords)
                .color_scale(ColorScale::Palette(ColorScalePalette::Viridis));

            let title = if i < 2 {
                format!("{} Solution - {:.0} Hz (dB SPL)", name, freq)
            } else {
                format!("{} - {:.0} Hz (Linear Scale)", name, freq)
            };

            let layout = Layout::new()
                .title(Title::from(title))
                .x_axis(Axis::new().title("X (m)").range(vec![0.0, config.width]))
                .y_axis(Axis::new().title("Y (m)").range(vec![0.0, config.depth]))
                .height(500)
                .width(600);

            let mut plot = Plot::new();
            plot.add_trace(trace);
            plot.set_layout(layout);

            let filename = format!(
                "plots/comparison_{}_{:.0}hz.html",
                name.to_lowercase()
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("-", ""),
                freq
            );
            plot.write_html(&filename);
            println!(
                "  Saved {} comparison plot: {}",
                name.to_lowercase(),
                filename
            );
        }
    }
}

fn main() {
    // Parse command-line arguments
    let args = Args::parse();

    // Rayon threading is used internally in FFT operations for large grids
    // Set number of threads from environment variable or use all available cores
    if let Ok(num_threads) = env::var("RAYON_NUM_THREADS") {
        if let Ok(n) = num_threads.parse::<usize>() {
            println!("Rayon thread pool: {} threads", n);
        }
    } else {
        // Use all available cores by default
        let num_cores = num_cpus::get();
        println!(
            "Rayon thread pool: {} threads (all available cores)",
            num_cores
        );
    }
    println!("Note: FFT operations within solver are parallelized automatically\n");

    // Create configuration from arguments
    let config = RoomConfig::from_args(&args);

    let mode_str = match config.mode {
        SimulationMode::TwoD => "2D",
        SimulationMode::ThreeD => "3D",
    };

    println!("\n{} Room Acoustics - Optimized Simulation", mode_str);
    println!("=========================================\n");

    // Display configuration
    println!("Room Configuration:");
    let shape_str = match config.shape {
        RoomShape::Rectangle => "Rectangle",
        RoomShape::Circle => "Circle",
        RoomShape::Sphere => "Sphere",
    };
    match config.mode {
        SimulationMode::TwoD => {
            println!("  Mode: 2D");
            println!("  Shape: {}", shape_str);
            println!(
                "  Room dimensions: {:.2}m (width) x {:.2}m (depth)",
                config.width, config.depth
            );
            println!("  Number of sources: {}", config.source_positions.len());
            for (i, pos) in config.source_positions.iter().enumerate() {
                println!("    Source {}: ({:.2}, {:.2})m", i + 1, pos.0, pos.1);
            }
            println!(
                "  Measurement point: ({:.2}, {:.2})m",
                config.measure_pos.0, config.measure_pos.1
            );
        }
        SimulationMode::ThreeD => {
            println!("  Mode: 3D");
            println!("  Shape: {}", shape_str);
            println!(
                "  Room dimensions: {:.2}m (width) x {:.2}m (depth) x {:.2}m (height)",
                config.width, config.depth, config.height
            );
            println!("  Number of sources: {}", config.source_positions.len());
            for (i, pos) in config.source_positions.iter().enumerate() {
                println!(
                    "    Source {}: ({:.2}, {:.2}, {:.2})m",
                    i + 1,
                    pos.0,
                    pos.1,
                    pos.2
                );
            }
            println!(
                "  Measurement point: ({:.2}, {:.2}, {:.2})m",
                config.measure_pos.0, config.measure_pos.1, config.measure_pos.2
            );
        }
    }
    println!("  Wall absorption: {:.1}%", config.absorption * 100.0);
    println!(
        "  Source SPL: {:.1} dB (pressure: {:.2e} Pa)",
        config.source_spl,
        spl_to_pressure(config.source_spl)
    );
    println!(
        "  Max grid resolution: {} points/m ({:.1} cm spacing)",
        config.max_grid_resolution,
        100.0 / config.max_grid_resolution as f64
    );
    println!("  Note: Actual resolution will be optimized per frequency");

    // Check analytical comparison requirements
    if config.compare_analytical {
        if config.shape != RoomShape::Rectangle {
            println!("\n⚠️  WARNING: Analytical comparison only available for rectangular rooms!");
            println!(
                "    Current shape: {:?}. Analytical comparison will be skipped.\n",
                config.shape
            );
        } else {
            println!("\n✅ Analytical comparison enabled for rectangular room");
            println!("    Using Neumann boundary conditions (rigid walls)");
            println!(
                "    Max modes: [5, 5, {}]\n",
                if config.mode == SimulationMode::TwoD {
                    "1"
                } else {
                    "5"
                }
            );
        }
    }

    println!("\nFrequency Analysis:");
    println!("  Range: {:.1} - {:.1} Hz", args.min_freq, args.max_freq);
    println!("  Number of points: {}\n", args.nb_freq);

    // Check mesh resolution adequacy and show optimization info
    let c_sound = 343.0;
    let min_wavelength = c_sound / args.max_freq;
    let max_pixel_size = 1.0 / config.max_grid_resolution as f64;
    let points_per_min_wavelength = min_wavelength / max_pixel_size;

    println!("Mesh Resolution Check:");
    println!(
        "  Minimum wavelength (at {:.1} Hz): {:.3} m",
        args.max_freq, min_wavelength
    );
    println!(
        "  Max pixel size: {:.3} m ({:.1} cm)",
        max_pixel_size,
        max_pixel_size * 100.0
    );
    println!(
        "  Points per minimum wavelength (at max resolution): {:.1}",
        points_per_min_wavelength
    );

    if points_per_min_wavelength < 4.0 {
        println!("  ❌ INSUFFICIENT: Need at least 4 points per wavelength (ideally 8+)");
        println!(
            "     Recommended minimum grid resolution: {:.0} points/m\n",
            (c_sound / args.max_freq) / max_pixel_size * 4.0 / points_per_min_wavelength
        );
    } else if points_per_min_wavelength < 8.0 {
        println!("  ⚠️  MARGINAL: Grid will be optimized per frequency");
        println!(
            "     Consider increasing --grid-resolution to {:.0} for best accuracy\n",
            config.max_grid_resolution as f64 * 8.0 / points_per_min_wavelength
        );
    } else {
        println!("  ✅ GOOD: Grid will be automatically optimized per frequency\n");
    }

    // Create plots directory
    std::fs::create_dir_all("plots").unwrap_or(());

    // Generate logarithmically spaced frequencies
    let frequencies = if args.nb_freq == 1 {
        vec![args.min_freq]
    } else {
        let log_min = args.min_freq.ln();
        let log_max = args.max_freq.ln();
        let log_step = (log_max - log_min) / (args.nb_freq - 1) as f64;

        (0..args.nb_freq)
            .map(|i| (log_min + i as f64 * log_step).exp())
            .collect()
    };

    // Update console header based on analytical comparison
    if config.compare_analytical && config.shape == RoomShape::Rectangle {
        println!(
            "Frequency (Hz) | SPL (dB) | Phase (°) | Resolution | Grid Size | Subdomains | Iterations | L2 Error | Max Error"
        );
        println!(
            "---------------|----------|-----------|------------|-----------|------------|------------|----------|----------"
        );
    } else {
        println!(
            "Frequency (Hz) | SPL (dB) | Phase (°) | Resolution | Grid Size | Subdomains | Iterations"
        );
        println!(
            "---------------|----------|-----------|------------|-----------|------------|------------"
        );
    }

    // Process frequencies sequentially, but use domain decomposition within each solve
    let mut spl_values = vec![];
    let mut phase_values = vec![];
    let mut field_data = vec![]; // Store fields for multi-frequency plot
    let mut error_metrics = vec![]; // Store error metrics for analytical comparison

    for &freq in &frequencies {
        // Get full field for plotting
        let (field, iters, grid_res, subdomains, pixel_size) =
            solve_frequency_full_field(&config, freq, args.num_subdomains);

        // Extract field at measurement point
        let measure_grid = (
            (config.measure_pos.0 / pixel_size) as usize,
            (config.measure_pos.1 / pixel_size) as usize,
            if config.mode == SimulationMode::TwoD {
                0
            } else {
                (config.measure_pos.2 / pixel_size) as usize
            },
        );
        let field_value = field.data[[measure_grid.0, measure_grid.1, measure_grid.2]];

        let spl = pressure_to_spl(field_value);
        let phase = field_value.arg() * 180.0 / PI;

        spl_values.push(spl);
        phase_values.push(phase);

        // Calculate grid size for display
        let nx = (config.width * grid_res as f64) as usize;
        let ny = (config.depth * grid_res as f64) as usize;
        let nz = match config.mode {
            SimulationMode::TwoD => 1,
            SimulationMode::ThreeD => (config.height * grid_res as f64) as usize,
        };
        let total_cells = nx * ny * nz;

        // Calculate z_slice for plotting
        let nz_field = field.data.shape()[2];
        let z_slice = if config.mode == SimulationMode::TwoD {
            0
        } else {
            nz_field / 2
        };

        // Perform analytical comparison if enabled
        let analytical_comparison =
            if config.compare_analytical && config.shape == RoomShape::Rectangle {
                if let Some(analytical_field) = analytical_room::compute_analytical_solution(
                    &config,
                    freq,
                    pixel_size,
                    (nx, ny, nz),
                ) {
                    let comparison = analytical_room::compare_solutions(&field, &analytical_field);

                    // Create error field for visualization
                    let mut error_field = field.clone();
                    for (num_val, ana_val) in error_field
                        .data
                        .iter_mut()
                        .zip(analytical_field.data.iter())
                    {
                        *num_val = *num_val - *ana_val;
                    }

                    // Generate comparison plots
                    analytical_room::plot_comparison(
                        &field,
                        &analytical_field,
                        &error_field,
                        pixel_size,
                        freq,
                        z_slice,
                        &config,
                    );

                    Some((
                        comparison.l2_error,
                        comparison.max_error,
                        comparison.relative_l2_error,
                    ))
                } else {
                    None
                }
            } else {
                None
            };

        // Store error metrics
        if let Some((l2_err, max_err, rel_err)) = analytical_comparison {
            error_metrics.push((freq, l2_err, max_err, rel_err));
        }

        // Print results with or without analytical comparison
        if config.compare_analytical && config.shape == RoomShape::Rectangle {
            if let Some((l2_err, max_err, _rel_err)) = analytical_comparison {
                println!(
                    "{:14.1} | {:8.2} | {:9.1} | {:6} pt/m | {:9} | {:4}x{:1}x{:1} | {:10} | {:8.2e} | {:8.2e}",
                    freq, spl, phase, grid_res, total_cells,
                    subdomains.0, subdomains.1, subdomains.2, iters,
                    l2_err, max_err
                );
            } else {
                println!(
                    "{:14.1} | {:8.2} | {:9.1} | {:6} pt/m | {:9} | {:4}x{:1}x{:1} | {:10} | {:>8} | {:>8}",
                    freq, spl, phase, grid_res, total_cells,
                    subdomains.0, subdomains.1, subdomains.2, iters,
                    "N/A", "N/A"
                );
            }
        } else {
            println!(
                "{:14.1} | {:8.2} | {:9.1} | {:6} pt/m | {:9} | {:4}x{:1}x{:1} | {:10}",
                freq,
                spl,
                phase,
                grid_res,
                total_cells,
                subdomains.0,
                subdomains.1,
                subdomains.2,
                iters
            );
        }

        // Store field for multi-frequency plot
        let nz = field.data.shape()[2];
        let z_slice = if config.mode == SimulationMode::TwoD {
            0
        } else {
            nz / 2
        };
        field_data.push((field, freq, pixel_size, z_slice));
    }

    // Combined plot with normalized dual y-axes for SPL and Phase
    // Since plotly-rs has limited dual-axis support, we'll normalize the values
    // and use a single plot with both traces

    // Normalize phase values to match SPL scale for visibility
    let phase_min = -180.0;
    let phase_max = 180.0;
    let spl_max = spl_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 5.0;
    let spl_min = spl_max - 50.0;

    // Create normalized phase values for display on same axis
    let phase_normalized: Vec<f64> = phase_values
        .iter()
        .map(|&p| spl_min + (p - phase_min) * (spl_max - spl_min) / (phase_max - phase_min))
        .collect();

    let spl_trace = Scatter::new(frequencies.clone(), spl_values.clone())
        .mode(Mode::LinesMarkers)
        .name(format!("SPL (dB) [Range: {:.1}-{:.1}]", spl_min, spl_max))
        .line(plotly::common::Line::new().color("blue").width(2.0));

    let phase_trace = Scatter::new(frequencies.clone(), phase_normalized)
        .mode(Mode::LinesMarkers)
        .name(format!(
            "Phase (°) [Range: {:.1}-{:.1}°]",
            phase_min, phase_max
        ))
        .line(
            plotly::common::Line::new()
                .color("red")
                .width(2.0)
                .dash(plotly::common::DashType::Dash),
        );

    let dual_layout = Layout::new()
        .title(Title::from(
            "Room Acoustic Frequency Response (SPL & Phase)",
        ))
        .x_axis(
            Axis::new()
                .title("Frequency (Hz)")
                .type_(plotly::layout::AxisType::Log)
                .grid_color("lightgray"),
        )
        .y_axis(Axis::new().title("SPL (dB) / Normalized Phase"))
        .height(650)
        .width(1024)
        .show_legend(true)
        .margin(Margin::new().bottom(60))
        .legend(
            Legend::new()
                .x_anchor(plotly::common::Anchor::Center)
                .x(0.5)
                .y_anchor(plotly::common::Anchor::Bottom)
                .y(-0.4)
                .orientation(plotly::common::Orientation::Horizontal),
        );

    let mut dual_plot = Plot::new();
    dual_plot.add_trace(spl_trace);
    dual_plot.add_trace(phase_trace);
    dual_plot.set_layout(dual_layout);
    dual_plot.write_html("plots/room_response_combined.html");

    println!("\nSaved combined SPL & Phase plot to plots/room_response_combined.html");

    // Plot all frequency fields in a single HTML file
    if !field_data.is_empty() {
        println!(
            "\nGenerating multi-frequency field plot for {} frequencies...",
            field_data.len()
        );
        plot_fields_multi_frequency(
            &field_data,
            config.mode,
            config.shape,
            config.width,
            config.depth,
            "plots/room_fields_all_frequencies.html",
        );
        println!("Saved multi-frequency field plot to plots/room_fields_all_frequencies.html");
    }

    // Plot analytical comparison summary if available
    if !error_metrics.is_empty() {
        println!("\nGenerating analytical comparison summary...");

        let freqs: Vec<f64> = error_metrics.iter().map(|(f, _, _, _)| *f).collect();
        let l2_errors: Vec<f64> = error_metrics.iter().map(|(_, l2, _, _)| *l2).collect();
        let max_errors: Vec<f64> = error_metrics.iter().map(|(_, _, max, _)| *max).collect();
        let rel_errors: Vec<f64> = error_metrics.iter().map(|(_, _, _, rel)| *rel).collect();

        // Plot L2 error vs frequency
        let l2_trace = Scatter::new(freqs.clone(), l2_errors.clone())
            .mode(Mode::LinesMarkers)
            .name("L2 Error")
            .line(plotly::common::Line::new().color("blue").width(2.0));

        let max_trace = Scatter::new(freqs.clone(), max_errors.clone())
            .mode(Mode::LinesMarkers)
            .name("Max Error")
            .line(
                plotly::common::Line::new()
                    .color("red")
                    .width(2.0)
                    .dash(plotly::common::DashType::Dash),
            );

        let error_layout = Layout::new()
            .title(Title::from("Analytical Comparison - Error vs Frequency"))
            .x_axis(
                Axis::new()
                    .title("Frequency (Hz)")
                    .type_(plotly::layout::AxisType::Log)
                    .grid_color("lightgray"),
            )
            .y_axis(
                Axis::new()
                    .title("Error")
                    .type_(plotly::layout::AxisType::Log),
            )
            .height(500)
            .width(800);

        let mut error_plot = Plot::new();
        error_plot.add_trace(l2_trace);
        error_plot.add_trace(max_trace);
        error_plot.set_layout(error_layout);
        error_plot.write_html("plots/analytical_error_summary.html");

        println!("  Saved error summary plot: plots/analytical_error_summary.html");

        // Print summary statistics
        let avg_l2_error = l2_errors.iter().sum::<f64>() / l2_errors.len() as f64;
        let avg_max_error = max_errors.iter().sum::<f64>() / max_errors.len() as f64;
        let avg_rel_error = rel_errors.iter().sum::<f64>() / rel_errors.len() as f64;

        println!("\nAnalytical Comparison Summary:");
        println!("  Average L2 error: {:.2e}", avg_l2_error);
        println!("  Average max error: {:.2e}", avg_max_error);
        println!("  Average relative error: {:.2e}", avg_rel_error);
        println!("  Number of frequencies compared: {}", error_metrics.len());
    }

    // Calculate theoretical room modes
    println!("\nTheoretical Room Modes:");
    let c = 343.0;
    match config.mode {
        SimulationMode::TwoD => {
            for nx in 1..=3 {
                for ny in 1..=3 {
                    let f = (c / 2.0)
                        * ((nx as f64 / config.width).powi(2) + (ny as f64 / config.depth).powi(2))
                            .sqrt();
                    if f <= 2000.0 {
                        println!("  Mode ({},{}): {:.1} Hz", nx, ny, f);
                    }
                }
            }
        }
        SimulationMode::ThreeD => {
            for nx in 1..=2 {
                for ny in 1..=2 {
                    for nz in 1..=2 {
                        let f = (c / 2.0)
                            * ((nx as f64 / config.width).powi(2)
                                + (ny as f64 / config.depth).powi(2)
                                + (nz as f64 / config.height).powi(2))
                            .sqrt();
                        if f <= 2000.0 {
                            println!("  Mode ({},{},{}): {:.1} Hz", nx, ny, nz, f);
                        }
                    }
                }
            }
        }
    }

    println!("\nPlots saved to 'plots/' directory!");

    // Show usage examples including analytical comparison
    if config.compare_analytical && !error_metrics.is_empty() {
        println!("\nAnalytical comparison plots generated:");
        println!("  - Individual frequency comparisons: plots/comparison_*_<freq>hz.html");
        println!("  - Error summary: plots/analytical_error_summary.html");
    }
}
