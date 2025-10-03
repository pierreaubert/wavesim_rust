//! Maxwell equation solver using FDTD (Finite-Difference Time-Domain) method
//!
//! This module implements a full 3D Maxwell solver for electromagnetic wave propagation

use crate::engine::array::{Complex64, WaveArray};
use num_complex::Complex;
use std::f64::consts::PI;

/// Maxwell domain for FDTD simulations
#[derive(Debug, Clone)]
pub struct MaxwellDomain {
    /// Electric permittivity (ε)
    pub permittivity: WaveArray<Complex64>,
    /// Magnetic permeability (μ)
    pub permeability: WaveArray<Complex64>,
    /// Conductivity (σ) for lossy media
    pub conductivity: WaveArray<Complex64>,
    /// Grid spacing
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    /// Time step (computed from CFL condition)
    pub dt: f64,
    /// Domain shape
    pub shape: (usize, usize, usize),
    /// Boundary conditions
    pub periodic: [bool; 3],
    /// PML (Perfectly Matched Layer) parameters
    pub pml_thickness: [usize; 3],
}

/// Field components for Maxwell equations
#[derive(Debug, Clone)]
pub struct ElectromagneticFields {
    /// Electric field components
    pub ex: WaveArray<Complex64>,
    pub ey: WaveArray<Complex64>,
    pub ez: WaveArray<Complex64>,
    /// Magnetic field components
    pub hx: WaveArray<Complex64>,
    pub hy: WaveArray<Complex64>,
    pub hz: WaveArray<Complex64>,
}

impl MaxwellDomain {
    /// Create a new Maxwell domain
    pub fn new(
        permittivity: WaveArray<Complex64>,
        permeability: WaveArray<Complex64>,
        dx: f64,
        dy: f64,
        dz: f64,
        periodic: [bool; 3],
        pml_thickness: [usize; 3],
    ) -> Self {
        let shape = permittivity.shape_tuple();

        // Compute time step from CFL condition
        // dt ≤ 1/(c * sqrt(1/dx² + 1/dy² + 1/dz²))
        let c = 299792458.0; // Speed of light in vacuum (m/s)
        let dt_max = 1.0 / (c * ((1.0 / dx.powi(2) + 1.0 / dy.powi(2) + 1.0 / dz.powi(2)).sqrt()));
        let dt = 0.99 * dt_max; // Use 99% of maximum for stability

        // Initialize conductivity to zero
        let conductivity = WaveArray::zeros(shape);

        Self {
            permittivity,
            permeability,
            conductivity,
            dx,
            dy,
            dz,
            dt,
            shape,
            periodic,
            pml_thickness,
        }
    }

    /// Set conductivity for lossy media
    pub fn with_conductivity(mut self, conductivity: WaveArray<Complex64>) -> Self {
        self.conductivity = conductivity;
        self
    }

    /// Initialize electromagnetic fields
    pub fn init_fields(&self) -> ElectromagneticFields {
        ElectromagneticFields {
            ex: WaveArray::zeros(self.shape),
            ey: WaveArray::zeros(self.shape),
            ez: WaveArray::zeros(self.shape),
            hx: WaveArray::zeros(self.shape),
            hy: WaveArray::zeros(self.shape),
            hz: WaveArray::zeros(self.shape),
        }
    }

    /// Update electric field components (E-field update)
    pub fn update_e_field(&self, fields: &mut ElectromagneticFields) {
        let shape = self.shape;

        // Update Ex
        for i in 1..shape.0 {
            for j in 1..shape.1 {
                for k in 1..shape.2 {
                    let curl_h = (fields.hz.data[[i, j, k]] - fields.hz.data[[i, j - 1, k]])
                        / self.dy
                        - (fields.hy.data[[i, j, k]] - fields.hy.data[[i, j, k - 1]]) / self.dz;

                    // Update with loss term
                    let sigma = self.conductivity.data[[i, j, k]];
                    let eps = self.permittivity.data[[i, j, k]];

                    fields.ex.data[[i, j, k]] = (Complex::new(1.0, 0.0)
                        - sigma * self.dt / (2.0 * eps))
                        * fields.ex.data[[i, j, k]]
                        + self.dt / eps * curl_h;
                }
            }
        }

        // Update Ey
        for i in 1..shape.0 {
            for j in 1..shape.1 {
                for k in 1..shape.2 {
                    let curl_h = (fields.hx.data[[i, j, k]] - fields.hx.data[[i, j, k - 1]])
                        / self.dz
                        - (fields.hz.data[[i, j, k]] - fields.hz.data[[i - 1, j, k]]) / self.dx;

                    let sigma = self.conductivity.data[[i, j, k]];
                    let eps = self.permittivity.data[[i, j, k]];

                    fields.ey.data[[i, j, k]] = (Complex::new(1.0, 0.0)
                        - sigma * self.dt / (2.0 * eps))
                        * fields.ey.data[[i, j, k]]
                        + self.dt / eps * curl_h;
                }
            }
        }

        // Update Ez
        for i in 1..shape.0 {
            for j in 1..shape.1 {
                for k in 1..shape.2 {
                    let curl_h = (fields.hy.data[[i, j, k]] - fields.hy.data[[i - 1, j, k]])
                        / self.dx
                        - (fields.hx.data[[i, j, k]] - fields.hx.data[[i, j - 1, k]]) / self.dy;

                    let sigma = self.conductivity.data[[i, j, k]];
                    let eps = self.permittivity.data[[i, j, k]];

                    fields.ez.data[[i, j, k]] = (Complex::new(1.0, 0.0)
                        - sigma * self.dt / (2.0 * eps))
                        * fields.ez.data[[i, j, k]]
                        + self.dt / eps * curl_h;
                }
            }
        }

        // Apply boundary conditions
        self.apply_e_boundary_conditions(fields);
    }

    /// Update magnetic field components (H-field update)
    pub fn update_h_field(&self, fields: &mut ElectromagneticFields) {
        let shape = self.shape;

        // Update Hx
        for i in 0..shape.0 - 1 {
            for j in 0..shape.1 - 1 {
                for k in 0..shape.2 - 1 {
                    let curl_e = (fields.ez.data[[i, j + 1, k]] - fields.ez.data[[i, j, k]])
                        / self.dy
                        - (fields.ey.data[[i, j, k + 1]] - fields.ey.data[[i, j, k]]) / self.dz;

                    let mu = self.permeability.data[[i, j, k]];
                    fields.hx.data[[i, j, k]] -= self.dt / mu * curl_e;
                }
            }
        }

        // Update Hy
        for i in 0..shape.0 - 1 {
            for j in 0..shape.1 - 1 {
                for k in 0..shape.2 - 1 {
                    let curl_e = (fields.ex.data[[i, j, k + 1]] - fields.ex.data[[i, j, k]])
                        / self.dz
                        - (fields.ez.data[[i + 1, j, k]] - fields.ez.data[[i, j, k]]) / self.dx;

                    let mu = self.permeability.data[[i, j, k]];
                    fields.hy.data[[i, j, k]] -= self.dt / mu * curl_e;
                }
            }
        }

        // Update Hz
        for i in 0..shape.0 - 1 {
            for j in 0..shape.1 - 1 {
                for k in 0..shape.2 - 1 {
                    let curl_e = (fields.ey.data[[i + 1, j, k]] - fields.ey.data[[i, j, k]])
                        / self.dx
                        - (fields.ex.data[[i, j + 1, k]] - fields.ex.data[[i, j, k]]) / self.dy;

                    let mu = self.permeability.data[[i, j, k]];
                    fields.hz.data[[i, j, k]] -= self.dt / mu * curl_e;
                }
            }
        }

        // Apply boundary conditions
        self.apply_h_boundary_conditions(fields);
    }

    /// Apply boundary conditions for E-field
    fn apply_e_boundary_conditions(&self, fields: &mut ElectromagneticFields) {
        let shape = self.shape;

        // Periodic boundaries
        if self.periodic[0] {
            for j in 0..shape.1 {
                for k in 0..shape.2 {
                    fields.ex.data[[0, j, k]] = fields.ex.data[[shape.0 - 1, j, k]];
                    fields.ey.data[[0, j, k]] = fields.ey.data[[shape.0 - 1, j, k]];
                    fields.ez.data[[0, j, k]] = fields.ez.data[[shape.0 - 1, j, k]];
                }
            }
        }

        if self.periodic[1] {
            for i in 0..shape.0 {
                for k in 0..shape.2 {
                    fields.ex.data[[i, 0, k]] = fields.ex.data[[i, shape.1 - 1, k]];
                    fields.ey.data[[i, 0, k]] = fields.ey.data[[i, shape.1 - 1, k]];
                    fields.ez.data[[i, 0, k]] = fields.ez.data[[i, shape.1 - 1, k]];
                }
            }
        }

        if self.periodic[2] {
            for i in 0..shape.0 {
                for j in 0..shape.1 {
                    fields.ex.data[[i, j, 0]] = fields.ex.data[[i, j, shape.2 - 1]];
                    fields.ey.data[[i, j, 0]] = fields.ey.data[[i, j, shape.2 - 1]];
                    fields.ez.data[[i, j, 0]] = fields.ez.data[[i, j, shape.2 - 1]];
                }
            }
        }

        // PML boundaries (simplified - full PML implementation would be more complex)
        self.apply_pml_e(fields);
    }

    /// Apply boundary conditions for H-field
    fn apply_h_boundary_conditions(&self, fields: &mut ElectromagneticFields) {
        let shape = self.shape;

        // Periodic boundaries
        if self.periodic[0] {
            for j in 0..shape.1 {
                for k in 0..shape.2 {
                    fields.hx.data[[shape.0 - 1, j, k]] = fields.hx.data[[0, j, k]];
                    fields.hy.data[[shape.0 - 1, j, k]] = fields.hy.data[[0, j, k]];
                    fields.hz.data[[shape.0 - 1, j, k]] = fields.hz.data[[0, j, k]];
                }
            }
        }

        if self.periodic[1] {
            for i in 0..shape.0 {
                for k in 0..shape.2 {
                    fields.hx.data[[i, shape.1 - 1, k]] = fields.hx.data[[i, 0, k]];
                    fields.hy.data[[i, shape.1 - 1, k]] = fields.hy.data[[i, 0, k]];
                    fields.hz.data[[i, shape.1 - 1, k]] = fields.hz.data[[i, 0, k]];
                }
            }
        }

        if self.periodic[2] {
            for i in 0..shape.0 {
                for j in 0..shape.1 {
                    fields.hx.data[[i, j, shape.2 - 1]] = fields.hx.data[[i, j, 0]];
                    fields.hy.data[[i, j, shape.2 - 1]] = fields.hy.data[[i, j, 0]];
                    fields.hz.data[[i, j, shape.2 - 1]] = fields.hz.data[[i, j, 0]];
                }
            }
        }

        // PML boundaries
        self.apply_pml_h(fields);
    }

    /// Apply PML absorption for E-field (simplified)
    fn apply_pml_e(&self, fields: &mut ElectromagneticFields) {
        // Simplified PML - in practice would need split-field formulation
        let shape = self.shape;

        // X-direction PML
        if self.pml_thickness[0] > 0 {
            let thickness = self.pml_thickness[0];
            for t in 0..thickness {
                let sigma = (t as f64 / thickness as f64).powi(3);
                let factor = Complex::new((-sigma * self.dt).exp(), 0.0);

                // Left boundary
                for j in 0..shape.1 {
                    for k in 0..shape.2 {
                        fields.ex.data[[t, j, k]] *= factor;
                        fields.ey.data[[t, j, k]] *= factor;
                        fields.ez.data[[t, j, k]] *= factor;
                    }
                }

                // Right boundary
                for j in 0..shape.1 {
                    for k in 0..shape.2 {
                        let i = shape.0 - 1 - t;
                        fields.ex.data[[i, j, k]] *= factor;
                        fields.ey.data[[i, j, k]] *= factor;
                        fields.ez.data[[i, j, k]] *= factor;
                    }
                }
            }
        }

        // Similar for Y and Z directions...
    }

    /// Apply PML absorption for H-field (simplified)
    fn apply_pml_h(&self, _fields: &mut ElectromagneticFields) {
        // Similar to apply_pml_e
    }

    /// Time-step the simulation
    pub fn step(&self, fields: &mut ElectromagneticFields) {
        self.update_h_field(fields);
        self.update_e_field(fields);
    }

    /// Add a source to the fields
    pub fn add_source(
        &self,
        fields: &mut ElectromagneticFields,
        source: &MaxwellSource,
        time: f64,
    ) {
        match &source.source_type {
            MaxwellSourceType::PointDipole {
                position,
                orientation,
                amplitude,
            } => {
                let signal = amplitude * (2.0 * PI * source.frequency * time).sin();

                // Add to appropriate field component based on orientation
                match orientation {
                    Orientation::X => {
                        fields.ex.data[[position[0], position[1], position[2]]] += signal
                    }
                    Orientation::Y => {
                        fields.ey.data[[position[0], position[1], position[2]]] += signal
                    }
                    Orientation::Z => {
                        fields.ez.data[[position[0], position[1], position[2]]] += signal
                    }
                }
            }
            MaxwellSourceType::PlaneWave { .. } => {
                // Implement plane wave source
            }
            MaxwellSourceType::GaussianPulse { .. } => {
                // Implement Gaussian pulse source
            }
        }
    }
}

/// Source types for Maxwell simulations
#[derive(Debug, Clone)]
pub struct MaxwellSource {
    pub source_type: MaxwellSourceType,
    pub frequency: f64,
}

#[derive(Debug, Clone)]
pub enum MaxwellSourceType {
    PointDipole {
        position: [usize; 3],
        orientation: Orientation,
        amplitude: Complex64,
    },
    PlaneWave {
        direction: [f64; 3],
        polarization: [f64; 3],
        amplitude: Complex64,
    },
    GaussianPulse {
        position: [usize; 3],
        width: f64,
        amplitude: Complex64,
    },
}

#[derive(Debug, Clone)]
pub enum Orientation {
    X,
    Y,
    Z,
}

/// Run Maxwell FDTD simulation
pub fn simulate_maxwell(
    domain: MaxwellDomain,
    sources: Vec<MaxwellSource>,
    num_steps: usize,
    save_interval: Option<usize>,
) -> Vec<ElectromagneticFields> {
    let mut fields = domain.init_fields();
    let mut results = Vec::new();

    for step in 0..num_steps {
        let time = step as f64 * domain.dt;

        // Add sources
        for source in &sources {
            domain.add_source(&mut fields, source, time);
        }

        // Update fields
        domain.step(&mut fields);

        // Save results if needed
        if let Some(interval) = save_interval {
            if step % interval == 0 {
                results.push(fields.clone());
            }
        }
    }

    // Always save final result
    results.push(fields);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxwell_domain_creation() {
        let shape = (50, 50, 50);
        let permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
        let permeability = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

        let domain = MaxwellDomain::new(
            permittivity,
            permeability,
            1e-6, // 1 micrometer
            1e-6,
            1e-6,
            [false, false, false],
            [10, 10, 10],
        );

        assert_eq!(domain.shape, shape);
        assert!(domain.dt > 0.0);
    }

    #[test]
    fn test_field_initialization() {
        let shape = (10, 10, 10);
        let permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
        let permeability = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

        let domain = MaxwellDomain::new(
            permittivity,
            permeability,
            1e-6,
            1e-6,
            1e-6,
            [true, true, true],
            [0, 0, 0],
        );

        let fields = domain.init_fields();

        assert_eq!(fields.ex.shape(), &[10, 10, 10]);
        assert_eq!(fields.ey.shape(), &[10, 10, 10]);
        assert_eq!(fields.ez.shape(), &[10, 10, 10]);
        assert_eq!(fields.hx.shape(), &[10, 10, 10]);
        assert_eq!(fields.hy.shape(), &[10, 10, 10]);
        assert_eq!(fields.hz.shape(), &[10, 10, 10]);
    }
}
