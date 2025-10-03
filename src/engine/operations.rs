//! Array operations for wave simulations

use crate::engine::array::{Complex64, WaveArray};
use ndarray::{Array3, Zip};
use num_traits::Zero;
use rustfft::{num_complex::Complex, FftPlanner};

/// Perform element-wise multiplication
pub fn multiply(a: &WaveArray<Complex64>, b: &WaveArray<Complex64>) -> WaveArray<Complex64> {
    WaveArray {
        data: &a.data * &b.data,
    }
}

/// Perform element-wise division
pub fn divide(a: &WaveArray<Complex64>, b: &WaveArray<Complex64>) -> WaveArray<Complex64> {
    WaveArray {
        data: &a.data / &b.data,
    }
}

/// Scale an array by a complex scalar and add an offset
/// out = scale * input + offset
pub fn scale(
    scale: Complex64,
    input: &WaveArray<Complex64>,
    offset: Option<Complex64>,
    out: &mut WaveArray<Complex64>,
) {
    if let Some(off) = offset {
        Zip::from(&mut out.data)
            .and(&input.data)
            .for_each(|o, &i| *o = scale * i + off);
    } else {
        Zip::from(&mut out.data)
            .and(&input.data)
            .for_each(|o, &i| *o = scale * i);
    }
}

/// Mix two arrays: out = alpha * a + beta * b
pub fn mix(
    alpha: Complex64,
    a: &WaveArray<Complex64>,
    beta: Complex64,
    b: &WaveArray<Complex64>,
    out: &mut WaveArray<Complex64>,
) {
    Zip::from(&mut out.data)
        .and(&a.data)
        .and(&b.data)
        .for_each(|o, &a_val, &b_val| *o = alpha * a_val + beta * b_val);
}

/// Linear interpolation: out = a + weight * (b - a)
pub fn lerp(
    a: &WaveArray<Complex64>,
    b: &WaveArray<Complex64>,
    weight: &WaveArray<Complex64>,
    out: &mut WaveArray<Complex64>,
) {
    Zip::from(&mut out.data)
        .and(&a.data)
        .and(&b.data)
        .and(&weight.data)
        .for_each(|o, &a_val, &b_val, &w| *o = a_val + w * (b_val - a_val));
}

/// Copy data from one array to another
pub fn copy(source: &WaveArray<Complex64>, dest: &mut WaveArray<Complex64>) {
    dest.data.assign(&source.data);
}

/// Perform 3D FFT
pub fn fft_3d(input: &WaveArray<Complex64>, output: &mut WaveArray<Complex64>) {
    // For simplicity, we'll use rustfft for 1D FFTs along each axis
    // In production, we'd want to use a more optimized 3D FFT implementation

    let shape = input.shape();
    output.data.assign(&input.data);

    let mut planner = FftPlanner::new();

    // FFT along axis 0
    for j in 0..shape[1] {
        for k in 0..shape[2] {
            let mut slice: Vec<Complex<f64>> =
                (0..shape[0]).map(|i| output.data[[i, j, k]]).collect();

            let fft = planner.plan_fft_forward(shape[0]);
            fft.process(&mut slice);

            for (i, val) in slice.iter().enumerate() {
                output.data[[i, j, k]] = *val;
            }
        }
    }

    // FFT along axis 1
    for i in 0..shape[0] {
        for k in 0..shape[2] {
            let mut slice: Vec<Complex<f64>> =
                (0..shape[1]).map(|j| output.data[[i, j, k]]).collect();

            let fft = planner.plan_fft_forward(shape[1]);
            fft.process(&mut slice);

            for (j, val) in slice.iter().enumerate() {
                output.data[[i, j, k]] = *val;
            }
        }
    }

    // FFT along axis 2
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let mut slice: Vec<Complex<f64>> =
                (0..shape[2]).map(|k| output.data[[i, j, k]]).collect();

            let fft = planner.plan_fft_forward(shape[2]);
            fft.process(&mut slice);

            for (k, val) in slice.iter().enumerate() {
                output.data[[i, j, k]] = *val;
            }
        }
    }
}

/// Perform 3D inverse FFT
pub fn ifft_3d(input: &WaveArray<Complex64>, output: &mut WaveArray<Complex64>) {
    let shape = input.shape();
    output.data.assign(&input.data);

    let mut planner = FftPlanner::new();
    let normalization = 1.0 / (shape[0] * shape[1] * shape[2]) as f64;

    // IFFT along axis 0
    for j in 0..shape[1] {
        for k in 0..shape[2] {
            let mut slice: Vec<Complex<f64>> =
                (0..shape[0]).map(|i| output.data[[i, j, k]]).collect();

            let fft = planner.plan_fft_inverse(shape[0]);
            fft.process(&mut slice);

            for (i, val) in slice.iter().enumerate() {
                output.data[[i, j, k]] = *val;
            }
        }
    }

    // IFFT along axis 1
    for i in 0..shape[0] {
        for k in 0..shape[2] {
            let mut slice: Vec<Complex<f64>> =
                (0..shape[1]).map(|j| output.data[[i, j, k]]).collect();

            let fft = planner.plan_fft_inverse(shape[1]);
            fft.process(&mut slice);

            for (j, val) in slice.iter().enumerate() {
                output.data[[i, j, k]] = *val;
            }
        }
    }

    // IFFT along axis 2
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let mut slice: Vec<Complex<f64>> =
                (0..shape[2]).map(|k| output.data[[i, j, k]]).collect();

            let fft = planner.plan_fft_inverse(shape[2]);
            fft.process(&mut slice);

            for (k, val) in slice.iter().enumerate() {
                output.data[[i, j, k]] = *val * normalization;
            }
        }
    }
}

/// Matrix multiplication along specified axis
pub fn matmul(
    matrix: &Array3<Complex64>,
    x: &WaveArray<Complex64>,
    axis: usize,
    out: &mut WaveArray<Complex64>,
) {
    // This is a simplified version - in production we'd use BLAS
    let shape = x.shape();

    match axis {
        0 => {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    for i_out in 0..matrix.shape()[0] {
                        let mut sum = Complex64::zero();
                        for i_in in 0..matrix.shape()[1] {
                            sum += matrix[[i_out, i_in, 0]] * x.data[[i_in, j, k]];
                        }
                        out.data[[i_out, j, k]] = sum;
                    }
                }
            }
        }
        1 => {
            for i in 0..shape[0] {
                for k in 0..shape[2] {
                    for j_out in 0..matrix.shape()[0] {
                        let mut sum = Complex64::zero();
                        for j_in in 0..matrix.shape()[1] {
                            sum += matrix[[j_out, j_in, 0]] * x.data[[i, j_in, k]];
                        }
                        out.data[[i, j_out, k]] = sum;
                    }
                }
            }
        }
        2 => {
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    for k_out in 0..matrix.shape()[0] {
                        let mut sum = Complex64::zero();
                        for k_in in 0..matrix.shape()[1] {
                            sum += matrix[[k_out, k_in, 0]] * x.data[[i, j, k_in]];
                        }
                        out.data[[i, j, k_out]] = sum;
                    }
                }
            }
        }
        _ => panic!("Invalid axis"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_scale_operation() {
        let input = WaveArray::from_scalar((2, 2, 2), Complex64::new(1.0, 0.0));
        let mut output = WaveArray::zeros((2, 2, 2));

        scale(
            Complex64::new(2.0, 0.0),
            &input,
            Some(Complex64::new(1.0, 0.0)),
            &mut output,
        );

        assert_eq!(output.data[[0, 0, 0]], Complex64::new(3.0, 0.0));
    }

    #[test]
    fn test_mix_operation() {
        let a = WaveArray::from_scalar((2, 2, 2), Complex64::new(1.0, 0.0));
        let b = WaveArray::from_scalar((2, 2, 2), Complex64::new(2.0, 0.0));
        let mut output = WaveArray::zeros((2, 2, 2));

        mix(
            Complex64::new(2.0, 0.0),
            &a,
            Complex64::new(3.0, 0.0),
            &b,
            &mut output,
        );

        assert_eq!(output.data[[0, 0, 0]], Complex64::new(8.0, 0.0));
    }
}
