//! Domain sizing utilities for optimal performance
//!
//! This module provides utilities to calculate optimal domain sizes that
//! maximize performance on hardware-accelerated backends.

/// Round up to the nearest power of 2
pub fn next_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    if n.is_power_of_two() {
        return n;
    }
    1 << (usize::BITS - (n - 1).leading_zeros())
}

/// Round down to the nearest power of 2
pub fn prev_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    1 << (usize::BITS - 1 - n.leading_zeros())
}

/// Find the nearest power of 2 (rounds to closest)
pub fn nearest_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }

    let next = next_power_of_2(n);
    let prev = prev_power_of_2(n);

    if next - n <= n - prev {
        next
    } else {
        prev
    }
}

/// Calculate optimal domain size from physical dimensions
///
/// Given a physical size and desired resolution, this function calculates
/// an optimal grid size that is close to the desired resolution but uses
/// power-of-2 dimensions for maximum performance with the Accelerate backend.
///
/// # Arguments
/// * `physical_size` - Physical dimension in meters/micrometers
/// * `pixel_size` - Desired pixel/grid size in same units
/// * `prefer_larger` - If true, rounds up; if false, rounds to nearest
///
/// # Returns
/// Optimal grid size (power of 2)
///
/// # Example
/// ```
/// use wavesim::utilities::domain_sizing::optimal_grid_size;
///
/// // For a 10 μm domain with 0.125 μm pixels
/// // Naive: 10 / 0.125 = 80 pixels (not power-of-2)
/// // Optimal: rounds to 64 or 128
/// let size = optimal_grid_size(10.0, 0.125, false);
/// assert_eq!(size, 64); // Nearest power-of-2 to 80
/// ```
pub fn optimal_grid_size(physical_size: f64, pixel_size: f64, prefer_larger: bool) -> usize {
    let naive_size = (physical_size / pixel_size).round() as usize;

    if prefer_larger {
        next_power_of_2(naive_size)
    } else {
        nearest_power_of_2(naive_size)
    }
}

/// Calculate optimal domain shape from physical dimensions
///
/// # Arguments
/// * `physical_shape` - Physical dimensions [x, y, z] in consistent units
/// * `pixel_size` - Grid spacing in same units
/// * `prefer_larger` - If true, rounds up; if false, rounds to nearest
///
/// # Returns
/// Optimal grid shape [nx, ny, nz] with power-of-2 dimensions
///
/// # Example
/// ```
/// use wavesim::utilities::domain_sizing::optimal_domain_shape;
///
/// // For a 12×12×12 μm domain with 0.125 μm pixels
/// // Naive: [96, 96, 96] (not power-of-2)
/// // Optimal: [128, 128, 128] or [64, 64, 64]
/// let shape = optimal_domain_shape([12.0, 12.0, 12.0], 0.125, true);
/// assert_eq!(shape, [128, 128, 128]);
/// ```
pub fn optimal_domain_shape(
    physical_shape: [f64; 3],
    pixel_size: f64,
    prefer_larger: bool,
) -> [usize; 3] {
    [
        optimal_grid_size(physical_shape[0], pixel_size, prefer_larger),
        optimal_grid_size(physical_shape[1], pixel_size, prefer_larger),
        optimal_grid_size(physical_shape[2], pixel_size, prefer_larger),
    ]
}

/// Check if all dimensions are powers of 2
pub fn is_power_of_2_shape(shape: &[usize]) -> bool {
    shape.iter().all(|&dim| dim > 0 && dim.is_power_of_two())
}

/// Get the performance tier for a given shape
///
/// Returns:
/// - "optimal" if all dimensions are powers of 2
/// - "suboptimal" if some but not all dimensions are powers of 2
/// - "slow" if no dimensions are powers of 2
pub fn performance_tier(shape: &[usize]) -> &'static str {
    let pow2_count = shape
        .iter()
        .filter(|&&dim| dim > 0 && dim.is_power_of_two())
        .count();

    match pow2_count {
        n if n == shape.len() => "optimal",
        0 => "slow",
        _ => "suboptimal",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_power_of_2() {
        assert_eq!(next_power_of_2(0), 1);
        assert_eq!(next_power_of_2(1), 1);
        assert_eq!(next_power_of_2(2), 2);
        assert_eq!(next_power_of_2(3), 4);
        assert_eq!(next_power_of_2(5), 8);
        assert_eq!(next_power_of_2(63), 64);
        assert_eq!(next_power_of_2(64), 64);
        assert_eq!(next_power_of_2(65), 128);
    }

    #[test]
    fn test_prev_power_of_2() {
        assert_eq!(prev_power_of_2(0), 0);
        assert_eq!(prev_power_of_2(1), 1);
        assert_eq!(prev_power_of_2(2), 2);
        assert_eq!(prev_power_of_2(3), 2);
        assert_eq!(prev_power_of_2(5), 4);
        assert_eq!(prev_power_of_2(63), 32);
        assert_eq!(prev_power_of_2(64), 64);
        assert_eq!(prev_power_of_2(127), 64);
    }

    #[test]
    fn test_nearest_power_of_2() {
        assert_eq!(nearest_power_of_2(0), 0);
        assert_eq!(nearest_power_of_2(1), 1);
        assert_eq!(nearest_power_of_2(3), 4); // 4-3=1 < 3-2=1, so 4
        assert_eq!(nearest_power_of_2(5), 4); // 8-5=3 > 5-4=1, so 4
        assert_eq!(nearest_power_of_2(6), 8); // 8-6=2 <= 6-4=2, so 8
        assert_eq!(nearest_power_of_2(48), 64); // 64-48=16 < 48-32=16, so 64
        assert_eq!(nearest_power_of_2(80), 64); // 128-80=48 > 80-64=16, so 64
        assert_eq!(nearest_power_of_2(96), 128); // 128-96=32 = 96-64=32, so 128 (<=)
    }

    #[test]
    fn test_optimal_grid_size() {
        // 10 / 0.125 = 80, should round to 64
        assert_eq!(optimal_grid_size(10.0, 0.125, false), 64);

        // With prefer_larger, should round up to 128
        assert_eq!(optimal_grid_size(10.0, 0.125, true), 128);

        // 16 / 0.25 = 64, already power-of-2
        assert_eq!(optimal_grid_size(16.0, 0.25, false), 64);
    }

    #[test]
    fn test_optimal_domain_shape() {
        // 12×12×12 with pixel_size=0.125 → naive: [96, 96, 96]
        // With prefer_larger=true: rounds up to [128, 128, 128]
        let shape = optimal_domain_shape([12.0, 12.0, 12.0], 0.125, true);
        assert_eq!(shape, [128, 128, 128]);

        // With prefer_larger=false: 96 is equidistant from 64 and 128
        // Due to <= in comparison, it rounds up to 128
        let shape2 = optimal_domain_shape([12.0, 12.0, 12.0], 0.125, false);
        assert_eq!(shape2, [128, 128, 128]); // 128-96=32 = 96-64=32

        // Test a case that clearly rounds down
        // 10 / 0.125 = 80, which is closer to 64 than 128
        let shape3 = optimal_domain_shape([10.0, 10.0, 10.0], 0.125, false);
        assert_eq!(shape3, [64, 64, 64]); // 80-64=16 < 128-80=48
    }

    #[test]
    fn test_is_power_of_2_shape() {
        assert!(is_power_of_2_shape(&[64, 128, 256]));
        assert!(is_power_of_2_shape(&[1, 2, 4, 8]));
        assert!(!is_power_of_2_shape(&[64, 128, 96]));
        assert!(!is_power_of_2_shape(&[5, 10, 15]));
    }

    #[test]
    fn test_performance_tier() {
        assert_eq!(performance_tier(&[64, 128, 256]), "optimal");
        assert_eq!(performance_tier(&[64, 96, 128]), "suboptimal");
        assert_eq!(performance_tier(&[65, 97, 129]), "slow");
    }
}
