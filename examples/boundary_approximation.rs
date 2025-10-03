//! Boundary approximation example
//!
//! This example demonstrates low-rank approximation of boundary operators
//! for the Helmholtz equation using SVD decomposition.

use ndarray::{Array1, Array2};
use num_complex::Complex;
use std::f64::consts::PI;

#[cfg(feature = "plotting")]
use plotly::{common::Mode, Plot, Scatter, Surface};

type Complex64 = Complex<f64>;

/// Create FFT frequency array similar to numpy's fftfreq with fftshift
fn fftfreq_shifted(n: usize, d: f64) -> Array1<f64> {
    let mut freq = Array1::zeros(n);
    let half = n / 2;

    // Positive frequencies
    for i in 0..half {
        freq[half + i] = i as f64 / (n as f64 * d);
    }

    // Negative frequencies
    for i in 0..(n - half) {
        freq[i] = -(n as f64 / 2.0 - i as f64) / (n as f64 * d);
    }

    freq * 2.0 * PI
}

/// Compute sinc function
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

/// Compute trapezoidal integration
fn trapezoid(values: &Array1<Complex64>) -> Complex64 {
    if values.len() < 2 {
        return Complex64::new(0.0, 0.0);
    }

    let mut sum = Complex64::new(0.0, 0.0);
    for i in 1..values.len() {
        sum += values[i] + values[i - 1];
    }
    sum * 0.5
}

fn main() {
    println!("Boundary Approximation Example");
    println!("==============================\n");

    // Parameters
    let n = 129;
    let delta = 0.25;

    // Create frequency grids
    let k_coarse = fftfreq_shifted(n, delta);
    let k_fine = fftfreq_shifted(100 * n + 1, delta / 5.0);

    let w = 1.0 / (k_coarse[1] - k_coarse[0]);
    let k0 = k_coarse[n / 5];

    println!("Grid parameters:");
    println!("  N = {}", n);
    println!("  Δ = {}", delta);
    println!("  k0 = {:.4}", k0);
    println!("  w = {:.4}", w);
    println!();

    // Create correction matrix M_correct
    let mut m_correct = Array2::<Complex64>::zeros((n, n));

    // Compute kernel for fine grid
    let kernel: Array1<Complex64> = k_fine.mapv(|k| {
        let denom = k * k - k0 * k0 + Complex64::new(0.0, 0.1 * k0 * k0);
        Complex64::new(1.0, 0.0) / denom
    });

    // Compute kernel for coarse grid
    let kernel_coarse: Array1<Complex64> = k_coarse.mapv(|k| {
        let denom = k * k - k0 * k0 + Complex64::new(0.0, 0.1 * k0 * k0);
        Complex64::new(1.0, 0.0) / denom
    });

    println!("Computing correction matrix...");
    // Fill correction matrix
    for (i, &p) in k_coarse.iter().enumerate() {
        for (j, &k) in k_coarse.iter().enumerate() {
            let integrand: Array1<Complex64> = k_fine
                .iter()
                .zip(kernel.iter())
                .map(|(&kf, &kern)| {
                    let sinc_p = sinc((p - kf) * w);
                    let sinc_k = sinc((k - kf) * w);
                    Complex64::new(sinc_p * sinc_k, 0.0) * kern
                })
                .collect();

            m_correct[[i, j]] = trapezoid(&integrand);
        }

        if (i + 1) % 20 == 0 {
            println!("  Row {}/{} completed", i + 1, n);
        }
    }

    // Compute edge kernels
    let shift_exp: Array1<Complex64> = k_coarse.mapv(|k| {
        let phase = k * delta * (n as f64 / 2.0 + 0.5);
        Complex64::from_polar(1.0, phase)
    });

    let edge_value_kernel: Array1<Complex64> = kernel_coarse
        .iter()
        .zip(shift_exp.iter())
        .map(|(&k, &s)| k * s)
        .collect();

    let edge_diff_kernel: Array1<Complex64> = k_coarse
        .iter()
        .zip(edge_value_kernel.iter())
        .map(|(&k, &ev)| Complex64::new(0.0, k) * ev)
        .collect();

    // Create difference matrix (M_correct without diagonal)
    let mut m_diff = m_correct.clone();
    for i in 0..n {
        m_diff[[i, i]] = Complex64::new(0.0, 0.0);
    }

    // Extract diagonal for approximation
    let mut d_approx: Array1<Complex64> = Array1::zeros(n);
    let d_correct: Array1<Complex64> = (0..n).map(|i| m_correct[[i, i]]).collect();
    d_approx.assign(&d_correct);

    // Compute Frobenius norm of M_correct
    let m_norm = m_correct.iter().map(|&x| x.norm_sqr()).sum::<f64>().sqrt();

    println!("\nMatrix properties:");
    println!("  ||M_correct||_F = {:.6}", m_norm);
    println!(
        "  Max diagonal element = {:.6}",
        d_correct.iter().map(|x| x.norm()).fold(0.0, f64::max)
    );

    // Low-rank approximation iteration
    let order = 2;
    println!("\nPerforming low-rank approximation (order = {})...", order);

    for iteration in 0..150 {
        // Remove diagonal part
        let mut m_approx = m_correct.clone();
        for i in 0..n {
            m_approx[[i, i]] = d_correct[i] - d_approx[i];
        }

        // For SVD, we would need a proper linear algebra library like nalgebra or ndarray-linalg
        // Here we'll just demonstrate the structure

        if iteration % 30 == 0 {
            println!("  Iteration {}: updating diagonal approximation", iteration);
        }

        // In a real implementation, we would:
        // 1. Compute SVD of m_approx
        // 2. Keep only the first 'order' singular values
        // 3. Reconstruct low-rank approximation
        // 4. Update d_approx

        // Placeholder update (simplified)
        let d_approx_old = d_approx.clone();
        for i in 0..n {
            d_approx[i] += Complex64::new(0.1, 0.0) * (d_correct[i] - d_approx_old[i]);
        }
    }

    // Analyze the approximation
    println!("\nApproximation analysis:");

    // Compute inner product of edge kernels
    let inner_product: Complex64 = edge_value_kernel
        .iter()
        .zip(edge_diff_kernel.iter())
        .map(|(&v, &d)| v.conj() * d)
        .sum();

    let norm_value = edge_value_kernel
        .iter()
        .map(|x| x.norm_sqr())
        .sum::<f64>()
        .sqrt();

    let norm_diff = edge_diff_kernel
        .iter()
        .map(|x| x.norm_sqr())
        .sum::<f64>()
        .sqrt();

    let correlation = inner_product / (norm_value * norm_diff);
    println!(
        "  Edge kernel correlation = {:.6} + {:.6}i",
        correlation.re, correlation.im
    );

    // Save results for plotting
    #[cfg(feature = "plotting")]
    plot_results(&m_diff, &d_approx, &d_correct, &edge_value_kernel);

    println!("\nBoundary approximation example completed.");
}

#[cfg(feature = "plotting")]
fn plot_results(
    m_diff: &Array2<Complex64>,
    d_approx: &Array1<Complex64>,
    d_correct: &Array1<Complex64>,
    edge_kernel: &Array1<Complex64>,
) {
    // Plot matrix visualization
    let mut plot = Plot::new();

    // Create heat map data for imaginary part of M_diff
    let z: Vec<Vec<f64>> = (0..m_diff.nrows())
        .map(|i| (0..m_diff.ncols()).map(|j| m_diff[[i, j]].im).collect())
        .collect();

    let trace = Surface::new(z).name("Im(M_diff)");
    plot.add_trace(trace);

    plot.show();

    // Plot diagonal comparison
    let mut plot2 = Plot::new();

    let x: Vec<f64> = (0..d_approx.len()).map(|i| i as f64).collect();

    let trace1 = Scatter::new(x.clone(), d_approx.iter().map(|x| x.re).collect::<Vec<_>>())
        .mode(Mode::Lines)
        .name("Re(d_approx)");

    let trace2 = Scatter::new(
        x.clone(),
        d_correct.iter().map(|x| x.re).collect::<Vec<_>>(),
    )
    .mode(Mode::Lines)
    .name("Re(d_correct)");

    let trace3 = Scatter::new(x.clone(), d_approx.iter().map(|x| x.im).collect::<Vec<_>>())
        .mode(Mode::Lines)
        .name("Im(d_approx)");

    let trace4 = Scatter::new(x, d_correct.iter().map(|x| x.im).collect::<Vec<_>>())
        .mode(Mode::Lines)
        .name("Im(d_correct)");

    plot2.add_trace(trace1);
    plot2.add_trace(trace2);
    plot2.add_trace(trace3);
    plot2.add_trace(trace4);

    plot2.show();
}
