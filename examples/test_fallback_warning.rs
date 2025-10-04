//! Test example to demonstrate Accelerate backend fallback warning
//!
//! This example intentionally uses non-power-of-2 dimensions to trigger
//! the fallback warning when compiled with --features accelerate.
//!
//! Run with:
//! ```bash
//! cargo run --example test_fallback_warning --features accelerate
//! ```

use num_complex::Complex;
use wavesim::{
    domain::helmholtz::HelmholtzDomain,
    domain::iteration::{preconditioned_richardson, IterationConfig},
    engine::array::WaveArray,
    utilities::domain_sizing::{is_power_of_2_shape, optimal_domain_shape},
};

fn main() {
    println!("=== Accelerate Backend Fallback Warning Demo ===\n");

    // Example 1: Non-power-of-2 dimensions (will trigger warning)
    println!("Example 1: Non-optimal dimensions");
    println!("-----------------------------------");

    let bad_shape = (80, 80, 80);
    println!("Creating domain with shape: {:?}", bad_shape);
    println!(
        "Is power-of-2: {}\n",
        is_power_of_2_shape(&[bad_shape.0, bad_shape.1, bad_shape.2])
    );

    let permittivity_bad = WaveArray::from_scalar(bad_shape, Complex::new(1.0, 0.0));
    let source_bad = WaveArray::zeros(bad_shape);

    let domain_bad = HelmholtzDomain::new(
        permittivity_bad,
        0.125,
        1.0,
        [false, false, false],
        [[4, 4], [4, 4], [4, 4]],
    );

    let config = IterationConfig {
        max_iterations: 5, // Just a few iterations to trigger FFT
        threshold: 1e-6,
        alpha: 0.75,
        full_residuals: false,
    };

    println!("Running simulation (watch for warning above)...");
    let _result_bad = preconditioned_richardson(&domain_bad, &source_bad, config.clone());
    println!("Simulation completed.\n");

    // Example 2: Optimal power-of-2 dimensions (no warning)
    println!("\nExample 2: Optimal dimensions (using helper)");
    println!("---------------------------------------------");

    let physical_size = [10.0, 10.0, 10.0];
    let pixel_size = 0.125;

    println!("Physical size: {:?} μm", physical_size);
    println!("Pixel size: {} μm", pixel_size);

    let good_shape = optimal_domain_shape(physical_size, pixel_size, false);
    println!("Optimal shape: {:?}", good_shape);
    println!("Is power-of-2: {}\n", is_power_of_2_shape(&good_shape));

    let permittivity_good = WaveArray::from_scalar(
        (good_shape[0], good_shape[1], good_shape[2]),
        Complex::new(1.0, 0.0),
    );
    let source_good = WaveArray::zeros((good_shape[0], good_shape[1], good_shape[2]));

    let domain_good = HelmholtzDomain::new(
        permittivity_good,
        pixel_size,
        1.0,
        [false, false, false],
        [[4, 4], [4, 4], [4, 4]],
    );

    println!("Running optimized simulation (no warning expected)...");
    let _result_good = preconditioned_richardson(&domain_good, &source_good, config);
    println!("Simulation completed successfully!\n");

    println!("=== Summary ===");
    println!(
        "✅ Optimal shape {:?} runs 5-10x faster on macOS with Accelerate",
        good_shape
    );
    println!(
        "⚠️  Non-optimal shape {:?} falls back to RustFFT (slower)",
        bad_shape
    );
    println!(
        "\n💡 Always use utilities::domain_sizing::optimal_domain_shape() for best performance!"
    );
}
