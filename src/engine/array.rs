//! Core array type for wave simulations
//!
//! This module provides the fundamental array type used throughout the library.
//! It wraps ndarray for efficient numerical operations with complex numbers.

use ndarray::{Array3, ArrayView3, ArrayViewMut3};
use num_complex::Complex;
use num_traits::Zero;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// Type alias for Complex64
pub type Complex64 = Complex<f64>;

/// Type alias for Complex32
pub type Complex32 = Complex<f32>;

/// The main array type for wave simulations
#[derive(Debug, Clone)]
pub struct WaveArray<T = Complex64> {
    /// The underlying ndarray
    pub data: Array3<T>,
}

impl<T> WaveArray<T>
where
    T: Clone + Zero,
{
    /// Create a new array with zeros
    pub fn zeros(shape: (usize, usize, usize)) -> Self {
        Self {
            data: Array3::zeros(shape),
        }
    }

    /// Create a new array from a scalar value
    pub fn from_scalar(shape: (usize, usize, usize), value: T) -> Self {
        Self {
            data: Array3::from_elem(shape, value),
        }
    }

    /// Get the shape of the array
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Get the shape as a tuple
    pub fn shape_tuple(&self) -> (usize, usize, usize) {
        let shape = self.shape();
        (shape[0], shape[1], shape[2])
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Get a view of the array
    pub fn view(&self) -> ArrayView3<'_, T> {
        self.data.view()
    }

    /// Get a mutable view of the array
    pub fn view_mut(&mut self) -> ArrayViewMut3<'_, T> {
        self.data.view_mut()
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Clone the underlying data
    pub fn to_owned(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

impl WaveArray<Complex64> {
    /// Create a new complex array from real data
    pub fn from_real(real_data: Array3<f64>) -> Self {
        let shape = real_data.shape();
        let mut complex_data = Array3::<Complex64>::zeros((shape[0], shape[1], shape[2]));

        for ((i, j, k), &val) in real_data.indexed_iter() {
            complex_data[[i, j, k]] = Complex::new(val, 0.0);
        }

        Self { data: complex_data }
    }

    /// Convert array to complex if it isn't already
    pub fn as_complex(array: Array3<f64>) -> Self {
        Self::from_real(array)
    }

    /// Create array with specific dtype compatibility
    pub fn with_dtype(shape: (usize, usize, usize), _dtype_like: Option<&Self>) -> Self {
        Self::zeros(shape)
    }

    /// Fill the array with a scalar value
    pub fn fill(&mut self, value: Complex64) {
        self.data.fill(value);
    }

    /// Compute the norm squared of the array
    pub fn norm_squared(&self) -> f64 {
        self.data.iter().map(|&c| (c.conj() * c).re).sum()
    }

    /// Compute the inner product with another array
    pub fn inner_product(&self, other: &Self) -> Complex64 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a.conj() * b)
            .fold(Complex64::zero(), |acc, x| acc + x)
    }
}

// Implement basic arithmetic operations
impl Add for WaveArray<Complex64> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            data: &self.data + &other.data,
        }
    }
}

impl Sub for WaveArray<Complex64> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            data: &self.data - &other.data,
        }
    }
}

impl Mul<Complex64> for WaveArray<Complex64> {
    type Output = Self;

    fn mul(self, scalar: Complex64) -> Self {
        Self {
            data: &self.data * scalar,
        }
    }
}

impl AddAssign for WaveArray<Complex64> {
    fn add_assign(&mut self, other: Self) {
        self.data += &other.data;
    }
}

impl SubAssign for WaveArray<Complex64> {
    fn sub_assign(&mut self, other: Self) {
        self.data -= &other.data;
    }
}

impl MulAssign<Complex64> for WaveArray<Complex64> {
    fn mul_assign(&mut self, scalar: Complex64) {
        self.data *= scalar;
    }
}

/// Trait for array slicing operations
pub trait ArraySlice {
    /// Get a slice of the array
    fn slice(&self, start: [usize; 3], stop: [usize; 3]) -> Self;

    /// Get edges of specified width
    fn edges(&self, widths: &[[usize; 2]; 3]) -> Vec<Self>
    where
        Self: Sized;
}

impl ArraySlice for WaveArray<Complex64> {
    fn slice(&self, start: [usize; 3], stop: [usize; 3]) -> Self {
        let slice = self.data.slice(ndarray::s![
            start[0]..stop[0],
            start[1]..stop[1],
            start[2]..stop[2]
        ]);
        Self {
            data: slice.to_owned(),
        }
    }

    fn edges(&self, widths: &[[usize; 2]; 3]) -> Vec<Self> {
        let shape = self.shape();
        let mut edges = Vec::new();

        // For each dimension, extract the edges
        for dim in 0..3 {
            let (left_width, right_width) = (widths[dim][0], widths[dim][1]);

            if left_width > 0 {
                let start = [0, 0, 0];
                let mut stop = [shape[0], shape[1], shape[2]];
                stop[dim] = left_width;
                edges.push(self.slice(start, stop));
            }

            if right_width > 0 {
                let mut start = [0, 0, 0];
                let stop = [shape[0], shape[1], shape[2]];
                start[dim] = shape[dim] - right_width;
                edges.push(self.slice(start, stop));
            }
        }

        edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_array_creation() {
        let arr = WaveArray::<Complex64>::zeros((10, 10, 10));
        assert_eq!(arr.shape(), &[10, 10, 10]);
        assert_eq!(arr.len(), 1000);
    }

    #[test]
    fn test_array_arithmetic() {
        let mut arr1 = WaveArray::from_scalar((5, 5, 5), Complex64::new(1.0, 0.0));
        let arr2 = WaveArray::from_scalar((5, 5, 5), Complex64::new(2.0, 1.0));

        let arr3 = arr1.clone() + arr2.clone();
        assert_eq!(arr3.data[[0, 0, 0]], Complex64::new(3.0, 1.0));

        arr1 += arr2;
        assert_eq!(arr1.data[[0, 0, 0]], Complex64::new(3.0, 1.0));
    }

    #[test]
    fn test_norm_squared() {
        let arr = WaveArray::from_scalar((2, 2, 2), Complex64::new(1.0, 1.0));
        let norm_sq = arr.norm_squared();
        assert_abs_diff_eq!(norm_sq, 16.0, epsilon = 1e-10); // 8 elements * (1^2 + 1^2)
    }
}
