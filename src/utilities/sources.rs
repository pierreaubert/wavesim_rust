//! Advanced source types for wave simulations
//!
//! This module provides various source types including plane waves,
//! Gaussian beams, dipole sources, and more.

use crate::engine::array::{Complex64, WaveArray};
use ndarray::Array3;
use num_complex::Complex;
use std::f64::consts::PI;

/// Builder for creating different source types
#[derive(Debug, Clone)]
pub struct SourceBuilder {
    shape: (usize, usize, usize),
    pixel_size: f64,
    wavelength: f64,
}

impl SourceBuilder {
    /// Create a new source builder
    pub fn new(shape: (usize, usize, usize), pixel_size: f64, wavelength: f64) -> Self {
        Self {
            shape,
            pixel_size,
            wavelength,
        }
    }
    
    /// Create a point source
    pub fn point_source(
        &self,
        position: [usize; 3],
        amplitude: Complex64,
    ) -> (WaveArray<Complex64>, [usize; 3]) {
        let mut source = WaveArray::zeros((1, 1, 1));
        source.data[[0, 0, 0]] = amplitude;
        (source, position)
    }
    
    /// Create a plane wave source
    pub fn plane_wave(
        &self,
        source_plane: SourcePlane,
        theta: f64,  // Angle from normal in radians
        phi: f64,    // Azimuthal angle in radians
        amplitude: Complex64,
        position: Option<[usize; 3]>,
    ) -> (WaveArray<Complex64>, [usize; 3]) {
        let k = 2.0 * PI / self.wavelength;
        
        let (shape, pos) = match source_plane {
            SourcePlane::XY => {
                let shape = (self.shape.0, self.shape.1, 1);
                let pos = position.unwrap_or([0, 0, self.shape.2 / 2]);
                (shape, pos)
            }
            SourcePlane::XZ => {
                let shape = (self.shape.0, 1, self.shape.2);
                let pos = position.unwrap_or([0, self.shape.1 / 2, 0]);
                (shape, pos)
            }
            SourcePlane::YZ => {
                let shape = (1, self.shape.1, self.shape.2);
                let pos = position.unwrap_or([self.shape.0 / 2, 0, 0]);
                (shape, pos)
            }
        };
        
        let mut source = WaveArray::zeros(shape);
        
        // Calculate wave vector components
        let kx = k * theta.sin() * phi.cos();
        let ky = k * theta.sin() * phi.sin();
        let kz = k * theta.cos();
        
        // Fill source with plane wave
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                for k_idx in 0..shape.2 {
                    let x = i as f64 * self.pixel_size;
                    let y = j as f64 * self.pixel_size;
                    let z = k_idx as f64 * self.pixel_size;
                    
                    let phase = kx * x + ky * y + kz * z;
                    source.data[[i, j, k_idx]] = amplitude * Complex::new(phase.cos(), phase.sin());
                }
            }
        }
        
        (source, pos)
    }
    
    /// Create a Gaussian beam source
    pub fn gaussian_beam(
        &self,
        source_plane: SourcePlane,
        beam_waist: f64,  // Beam waist radius
        theta: f64,
        phi: f64,
        amplitude: Complex64,
        position: Option<[usize; 3]>,
    ) -> (WaveArray<Complex64>, [usize; 3]) {
        let k = 2.0 * PI / self.wavelength;
        
        let (shape, pos) = match source_plane {
            SourcePlane::XY => {
                let shape = (self.shape.0, self.shape.1, 1);
                let pos = position.unwrap_or([0, 0, self.shape.2 / 2]);
                (shape, pos)
            }
            SourcePlane::XZ => {
                let shape = (self.shape.0, 1, self.shape.2);
                let pos = position.unwrap_or([0, self.shape.1 / 2, 0]);
                (shape, pos)
            }
            SourcePlane::YZ => {
                let shape = (1, self.shape.1, self.shape.2);
                let pos = position.unwrap_or([self.shape.0 / 2, 0, 0]);
                (shape, pos)
            }
        };
        
        let mut source = WaveArray::zeros(shape);
        
        // Calculate wave vector components
        let kx = k * theta.sin() * phi.cos();
        let ky = k * theta.sin() * phi.sin();
        let kz = k * theta.cos();
        
        // Center of the beam
        let cx = shape.0 as f64 * self.pixel_size / 2.0;
        let cy = shape.1 as f64 * self.pixel_size / 2.0;
        
        // Fill source with Gaussian beam
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                for k_idx in 0..shape.2 {
                    let x = i as f64 * self.pixel_size;
                    let y = j as f64 * self.pixel_size;
                    let z = k_idx as f64 * self.pixel_size;
                    
                    // Gaussian envelope
                    let r2 = ((x - cx) / beam_waist).powi(2) + ((y - cy) / beam_waist).powi(2);
                    let envelope = (-r2).exp();
                    
                    // Phase
                    let phase = kx * x + ky * y + kz * z;
                    
                    source.data[[i, j, k_idx]] = 
                        amplitude * envelope * Complex::new(phase.cos(), phase.sin());
                }
            }
        }
        
        (source, pos)
    }
    
    /// Create a dipole source
    pub fn dipole_source(
        &self,
        position: [usize; 3],
        orientation: DipoleOrientation,
        amplitude: Complex64,
    ) -> (WaveArray<Complex64>, [usize; 3]) {
        // Dipole pattern extends over a small region
        let extent = 3; // 3x3x3 region
        let size = 2 * extent + 1;
        
        let mut source = WaveArray::zeros((size, size, size));
        
        for i in 0..size {
            for j in 0..size {
                for k in 0..size {
                    let di = i as f64 - extent as f64;
                    let dj = j as f64 - extent as f64;
                    let dk = k as f64 - extent as f64;
                    
                    let r = (di * di + dj * dj + dk * dk).sqrt();
                    if r < 1e-10 {
                        // At the center
                        source.data[[i, j, k]] = amplitude;
                    } else {
                        // Dipole radiation pattern
                        let pattern = match orientation {
                            DipoleOrientation::X => di.abs() / r,
                            DipoleOrientation::Y => dj.abs() / r,
                            DipoleOrientation::Z => dk.abs() / r,
                        };
                        
                        source.data[[i, j, k]] = amplitude * pattern * (-r * r / 4.0).exp();
                    }
                }
            }
        }
        
        // Adjust position to center the dipole
        let adjusted_pos = [
            position[0].saturating_sub(extent),
            position[1].saturating_sub(extent),
            position[2].saturating_sub(extent),
        ];
        
        (source, adjusted_pos)
    }
    
    /// Create a focused beam source (lens-focused)
    pub fn focused_beam(
        &self,
        focal_point: [f64; 3],
        numerical_aperture: f64,
        amplitude: Complex64,
        source_plane: SourcePlane,
    ) -> (WaveArray<Complex64>, [usize; 3]) {
        let k = 2.0 * PI / self.wavelength;
        
        let (shape, pos) = match source_plane {
            SourcePlane::XY => {
                let shape = (self.shape.0, self.shape.1, 1);
                let pos = [0, 0, 0];
                (shape, pos)
            }
            SourcePlane::XZ => {
                let shape = (self.shape.0, 1, self.shape.2);
                let pos = [0, 0, 0];
                (shape, pos)
            }
            SourcePlane::YZ => {
                let shape = (1, self.shape.1, self.shape.2);
                let pos = [0, 0, 0];
                (shape, pos)
            }
        };
        
        let mut source = WaveArray::zeros(shape);
        
        // Calculate the aperture angle
        let max_angle = numerical_aperture.asin();
        
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                for k_idx in 0..shape.2 {
                    let x = i as f64 * self.pixel_size;
                    let y = j as f64 * self.pixel_size;
                    let z = k_idx as f64 * self.pixel_size;
                    
                    // Distance to focal point
                    let dx = x - focal_point[0];
                    let dy = y - focal_point[1];
                    let dz = z - focal_point[2];
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();
                    
                    if r > 1e-10 {
                        // Check if within NA cone
                        let angle = (((dx * dx + dy * dy).sqrt()) / dz.abs()).atan();
                        
                        if angle <= max_angle {
                            // Spherical wave converging to focal point
                            let phase = k * r;
                            let amp = 1.0 / r.sqrt(); // Amplitude decreases with distance
                            
                            source.data[[i, j, k_idx]] = 
                                amplitude * amp * Complex::new((-phase).cos(), (-phase).sin());
                        }
                    }
                }
            }
        }
        
        (source, pos)
    }
    
    /// Create a vortex beam (orbital angular momentum)
    pub fn vortex_beam(
        &self,
        source_plane: SourcePlane,
        topological_charge: i32,
        beam_waist: f64,
        amplitude: Complex64,
        position: Option<[usize; 3]>,
    ) -> (WaveArray<Complex64>, [usize; 3]) {
        let (shape, pos) = match source_plane {
            SourcePlane::XY => {
                let shape = (self.shape.0, self.shape.1, 1);
                let pos = position.unwrap_or([0, 0, self.shape.2 / 2]);
                (shape, pos)
            }
            SourcePlane::XZ => {
                let shape = (self.shape.0, 1, self.shape.2);
                let pos = position.unwrap_or([0, self.shape.1 / 2, 0]);
                (shape, pos)
            }
            SourcePlane::YZ => {
                let shape = (1, self.shape.1, self.shape.2);
                let pos = position.unwrap_or([self.shape.0 / 2, 0, 0]);
                (shape, pos)
            }
        };
        
        let mut source = WaveArray::zeros(shape);
        
        // Center of the beam
        let cx = shape.0 as f64 * self.pixel_size / 2.0;
        let cy = shape.1 as f64 * self.pixel_size / 2.0;
        
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                for k_idx in 0..shape.2 {
                    let x = i as f64 * self.pixel_size - cx;
                    let y = j as f64 * self.pixel_size - cy;
                    
                    let r = (x * x + y * y).sqrt();
                    let theta = y.atan2(x);
                    
                    // Laguerre-Gaussian beam (simplified)
                    let gaussian = (-(r / beam_waist).powi(2)).exp();
                    let vortex_phase = topological_charge as f64 * theta;
                    
                    source.data[[i, j, k_idx]] = 
                        amplitude * gaussian * Complex::new(vortex_phase.cos(), vortex_phase.sin());
                }
            }
        }
        
        (source, pos)
    }
}

/// Source plane orientation
#[derive(Debug, Clone, Copy)]
pub enum SourcePlane {
    XY,
    XZ,
    YZ,
}

/// Dipole orientation
#[derive(Debug, Clone, Copy)]
pub enum DipoleOrientation {
    X,
    Y,
    Z,
}

/// Combined source for multiple simultaneous sources
pub struct MultiSource {
    sources: Vec<(WaveArray<Complex64>, [usize; 3])>,
}

impl MultiSource {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
        }
    }
    
    pub fn add_source(&mut self, source: WaveArray<Complex64>, position: [usize; 3]) {
        self.sources.push((source, position));
    }
    
    /// Combine all sources into a single array
    pub fn combine(&self, shape: (usize, usize, usize)) -> WaveArray<Complex64> {
        let mut combined = WaveArray::zeros(shape);
        
        for (source, pos) in &self.sources {
            let src_shape = source.shape();
            
            for i in 0..src_shape[0].min(shape.0 - pos[0]) {
                for j in 0..src_shape[1].min(shape.1 - pos[1]) {
                    for k in 0..src_shape[2].min(shape.2 - pos[2]) {
                        combined.data[[pos[0] + i, pos[1] + j, pos[2] + k]] += 
                            source.data[[i, j, k]];
                    }
                }
            }
        }
        
        combined
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_source_builder() {
        let builder = SourceBuilder::new((100, 100, 100), 0.1, 0.5);
        
        // Test point source
        let (source, pos) = builder.point_source([50, 50, 50], Complex::new(1.0, 0.0));
        assert_eq!(source.shape(), &[1, 1, 1]);
        assert_eq!(pos, [50, 50, 50]);
        
        // Test plane wave
        let (source, _) = builder.plane_wave(
            SourcePlane::XY,
            0.0,
            0.0,
            Complex::new(1.0, 0.0),
            None,
        );
        assert_eq!(source.shape()[2], 1);
    }
    
    #[test]
    fn test_multi_source() {
        let builder = SourceBuilder::new((100, 100, 100), 0.1, 0.5);
        let mut multi = MultiSource::new();
        
        let (s1, p1) = builder.point_source([10, 10, 10], Complex::new(1.0, 0.0));
        let (s2, p2) = builder.point_source([50, 50, 50], Complex::new(0.5, 0.0));
        
        multi.add_source(s1, p1);
        multi.add_source(s2, p2);
        
        let combined = multi.combine((100, 100, 100));
        
        // Check that sources were added
        assert_eq!(combined.data[[10, 10, 10]], Complex::new(1.0, 0.0));
        assert_eq!(combined.data[[50, 50, 50]], Complex::new(0.5, 0.0));
    }
}