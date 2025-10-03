use num_complex::Complex;
use wavesim::engine::array::{Complex64, WaveArray};
use wavesim::engine::operations::{fft_3d, ifft_3d};

#[test]
fn test_fft_roundtrip() {
    // Create a simple test array
    let mut input = WaveArray::zeros((4, 4, 4));
    input.data[[0, 0, 0]] = Complex::new(1.0, 0.0);
    input.data[[1, 1, 1]] = Complex::new(2.0, 0.0);

    println!("Input norm: {}", input.norm_squared());

    // FFT forward
    let mut fft_result = WaveArray::zeros((4, 4, 4));
    fft_3d(&input, &mut fft_result);
    println!("FFT result norm: {}", fft_result.norm_squared());

    // FFT inverse
    let mut recovered = WaveArray::zeros((4, 4, 4));
    ifft_3d(&fft_result, &mut recovered);
    println!("Recovered norm: {}", recovered.norm_squared());

    // Check values
    println!("Original [0,0,0]: {:?}", input.data[[0, 0, 0]]);
    println!("Recovered [0,0,0]: {:?}", recovered.data[[0, 0, 0]]);
    println!("Original [1,1,1]: {:?}", input.data[[1, 1, 1]]);
    println!("Recovered [1,1,1]: {:?}", recovered.data[[1, 1, 1]]);

    // Test that we get back approximately the same values
    let tolerance = 1e-10;
    assert!((recovered.data[[0, 0, 0]] - input.data[[0, 0, 0]]).norm() < tolerance);
    assert!((recovered.data[[1, 1, 1]] - input.data[[1, 1, 1]]).norm() < tolerance);
}
