//! Parallel Maxwell FDTD solver using domain decomposition
//!
//! This module implements a parallel version of the Maxwell FDTD solver
//! that uses domain decomposition with Rayon for shared-memory parallelization.

use crate::domain::maxwell::{ElectromagneticFields, MaxwellSource, Orientation};
use crate::domain_decomposition::{DomainDecomposition, Subdomain};
use crate::engine::array::{Complex64, WaveArray};
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

/// Parallel Maxwell domain for FDTD simulations with domain decomposition
#[derive(Debug, Clone)]
pub struct ParallelMaxwellDomain {
    /// Electric permittivity (ε)
    pub permittivity: Arc<WaveArray<Complex64>>,
    /// Magnetic permeability (μ)
    pub permeability: Arc<WaveArray<Complex64>>,
    /// Conductivity (σ) for lossy media
    pub conductivity: Arc<WaveArray<Complex64>>,
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
    /// Domain decomposition
    pub decomposition: Arc<DomainDecomposition>,
}

impl ParallelMaxwellDomain {
    /// Create a new parallel Maxwell domain
    pub fn new(
        permittivity: WaveArray<Complex64>,
        permeability: WaveArray<Complex64>,
        dx: f64,
        dy: f64,
        dz: f64,
        periodic: [bool; 3],
        pml_thickness: [usize; 3],
        num_subdomains: (usize, usize, usize),
    ) -> Self {
        let shape = permittivity.shape_tuple();

        // Compute time step from CFL condition
        let c = 299792458.0; // Speed of light in vacuum (m/s)
        let dt_max = 1.0 / (c * ((1.0 / dx.powi(2) + 1.0 / dy.powi(2) + 1.0 / dz.powi(2)).sqrt()));
        let dt = 0.99 * dt_max; // Use 99% of maximum for stability

        // Initialize conductivity to zero
        let conductivity = WaveArray::zeros(shape);

        // Create domain decomposition
        let decomposition = Arc::new(DomainDecomposition::new(shape, num_subdomains, 1));

        Self {
            permittivity: Arc::new(permittivity),
            permeability: Arc::new(permeability),
            conductivity: Arc::new(conductivity),
            dx,
            dy,
            dz,
            dt,
            shape,
            periodic,
            pml_thickness,
            decomposition,
        }
    }

    /// Set conductivity for lossy media
    pub fn with_conductivity(mut self, conductivity: WaveArray<Complex64>) -> Self {
        self.conductivity = Arc::new(conductivity);
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

    /// Update electric field components in parallel
    pub fn update_e_field_parallel(&self, fields: &mut ElectromagneticFields) {
        // Create thread-safe references to the fields
        let ex = Arc::new(Mutex::new(&mut fields.ex));
        let ey = Arc::new(Mutex::new(&mut fields.ey));
        let ez = Arc::new(Mutex::new(&mut fields.ez));
        let hx = Arc::new(&fields.hx);
        let hy = Arc::new(&fields.hy);
        let hz = Arc::new(&fields.hz);

        // Parallel update across subdomains
        self.decomposition
            .subdomains
            .par_iter()
            .for_each(|subdomain| {
                self.update_e_field_subdomain(
                    subdomain,
                    ex.clone(),
                    ey.clone(),
                    ez.clone(),
                    hx.clone(),
                    hy.clone(),
                    hz.clone(),
                );
            });

        // Apply boundary conditions
        self.apply_e_boundary_conditions(fields);
    }

    /// Update electric field for a subdomain
    fn update_e_field_subdomain(
        &self,
        subdomain: &Subdomain,
        ex: Arc<Mutex<&mut WaveArray<Complex64>>>,
        ey: Arc<Mutex<&mut WaveArray<Complex64>>>,
        ez: Arc<Mutex<&mut WaveArray<Complex64>>>,
        hx: Arc<&WaveArray<Complex64>>,
        hy: Arc<&WaveArray<Complex64>>,
        hz: Arc<&WaveArray<Complex64>>,
    ) {
        // Update Ex component
        for i in subdomain.start[0].max(1)..subdomain.end[0].min(self.shape.0) {
            for j in subdomain.start[1].max(1)..subdomain.end[1].min(self.shape.1) {
                for k in subdomain.start[2].max(1)..subdomain.end[2].min(self.shape.2) {
                    let curl_h = (hz.data[[i, j, k]] - hz.data[[i, j - 1, k]]) / self.dy
                        - (hy.data[[i, j, k]] - hy.data[[i, j, k - 1]]) / self.dz;

                    let sigma = self.conductivity.data[[i, j, k]];
                    let eps = self.permittivity.data[[i, j, k]];

                    let new_ex = (Complex::new(1.0, 0.0) - sigma * self.dt / (2.0 * eps))
                        * ex.lock().unwrap().data[[i, j, k]]
                        + self.dt / eps * curl_h;

                    ex.lock().unwrap().data[[i, j, k]] = new_ex;
                }
            }
        }

        // Similar updates for Ey and Ez components
        for i in subdomain.start[0].max(1)..subdomain.end[0].min(self.shape.0) {
            for j in subdomain.start[1].max(1)..subdomain.end[1].min(self.shape.1) {
                for k in subdomain.start[2].max(1)..subdomain.end[2].min(self.shape.2) {
                    let curl_h = (hx.data[[i, j, k]] - hx.data[[i, j, k - 1]]) / self.dz
                        - (hz.data[[i, j, k]] - hz.data[[i - 1, j, k]]) / self.dx;

                    let sigma = self.conductivity.data[[i, j, k]];
                    let eps = self.permittivity.data[[i, j, k]];

                    let new_ey = (Complex::new(1.0, 0.0) - sigma * self.dt / (2.0 * eps))
                        * ey.lock().unwrap().data[[i, j, k]]
                        + self.dt / eps * curl_h;

                    ey.lock().unwrap().data[[i, j, k]] = new_ey;
                }
            }
        }

        for i in subdomain.start[0].max(1)..subdomain.end[0].min(self.shape.0) {
            for j in subdomain.start[1].max(1)..subdomain.end[1].min(self.shape.1) {
                for k in subdomain.start[2].max(1)..subdomain.end[2].min(self.shape.2) {
                    let curl_h = (hy.data[[i, j, k]] - hy.data[[i - 1, j, k]]) / self.dx
                        - (hx.data[[i, j, k]] - hx.data[[i, j - 1, k]]) / self.dy;

                    let sigma = self.conductivity.data[[i, j, k]];
                    let eps = self.permittivity.data[[i, j, k]];

                    let new_ez = (Complex::new(1.0, 0.0) - sigma * self.dt / (2.0 * eps))
                        * ez.lock().unwrap().data[[i, j, k]]
                        + self.dt / eps * curl_h;

                    ez.lock().unwrap().data[[i, j, k]] = new_ez;
                }
            }
        }
    }

    /// Update magnetic field components in parallel
    pub fn update_h_field_parallel(&self, fields: &mut ElectromagneticFields) {
        // Create thread-safe references
        let ex = Arc::new(&fields.ex);
        let ey = Arc::new(&fields.ey);
        let ez = Arc::new(&fields.ez);
        let hx = Arc::new(Mutex::new(&mut fields.hx));
        let hy = Arc::new(Mutex::new(&mut fields.hy));
        let hz = Arc::new(Mutex::new(&mut fields.hz));

        // Parallel update across subdomains
        self.decomposition
            .subdomains
            .par_iter()
            .for_each(|subdomain| {
                self.update_h_field_subdomain(
                    subdomain,
                    ex.clone(),
                    ey.clone(),
                    ez.clone(),
                    hx.clone(),
                    hy.clone(),
                    hz.clone(),
                );
            });

        // Apply boundary conditions
        self.apply_h_boundary_conditions(fields);
    }

    /// Update magnetic field for a subdomain
    fn update_h_field_subdomain(
        &self,
        subdomain: &Subdomain,
        ex: Arc<&WaveArray<Complex64>>,
        ey: Arc<&WaveArray<Complex64>>,
        ez: Arc<&WaveArray<Complex64>>,
        hx: Arc<Mutex<&mut WaveArray<Complex64>>>,
        hy: Arc<Mutex<&mut WaveArray<Complex64>>>,
        hz: Arc<Mutex<&mut WaveArray<Complex64>>>,
    ) {
        // Update Hx component
        for i in subdomain.start[0]..subdomain.end[0].min(self.shape.0 - 1) {
            for j in subdomain.start[1]..subdomain.end[1].min(self.shape.1 - 1) {
                for k in subdomain.start[2]..subdomain.end[2].min(self.shape.2 - 1) {
                    let curl_e = (ez.data[[i, j + 1, k]] - ez.data[[i, j, k]]) / self.dy
                        - (ey.data[[i, j, k + 1]] - ey.data[[i, j, k]]) / self.dz;

                    let mu = self.permeability.data[[i, j, k]];
                    hx.lock().unwrap().data[[i, j, k]] -= self.dt / mu * curl_e;
                }
            }
        }

        // Update Hy component
        for i in subdomain.start[0]..subdomain.end[0].min(self.shape.0 - 1) {
            for j in subdomain.start[1]..subdomain.end[1].min(self.shape.1 - 1) {
                for k in subdomain.start[2]..subdomain.end[2].min(self.shape.2 - 1) {
                    let curl_e = (ex.data[[i, j, k + 1]] - ex.data[[i, j, k]]) / self.dz
                        - (ez.data[[i + 1, j, k]] - ez.data[[i, j, k]]) / self.dx;

                    let mu = self.permeability.data[[i, j, k]];
                    hy.lock().unwrap().data[[i, j, k]] -= self.dt / mu * curl_e;
                }
            }
        }

        // Update Hz component
        for i in subdomain.start[0]..subdomain.end[0].min(self.shape.0 - 1) {
            for j in subdomain.start[1]..subdomain.end[1].min(self.shape.1 - 1) {
                for k in subdomain.start[2]..subdomain.end[2].min(self.shape.2 - 1) {
                    let curl_e = (ey.data[[i + 1, j, k]] - ey.data[[i, j, k]]) / self.dx
                        - (ex.data[[i, j + 1, k]] - ex.data[[i, j, k]]) / self.dy;

                    let mu = self.permeability.data[[i, j, k]];
                    hz.lock().unwrap().data[[i, j, k]] -= self.dt / mu * curl_e;
                }
            }
        }
    }

    /// Apply boundary conditions for E-field (same as non-parallel version)
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

        // PML boundaries (simplified)
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
        let shape = self.shape;

        // X-direction PML
        if self.pml_thickness[0] > 0 {
            let thickness = self.pml_thickness[0];
            for t in 0..thickness {
                let sigma = (t as f64 / thickness as f64).powi(3);
                let factor = Complex::new((-sigma * self.dt).exp(), 0.0);

                // Apply PML without parallelization to avoid borrowing issues
                for j in 0..shape.1 {
                    for k in 0..shape.2 {
                        // Left boundary
                        fields.ex.data[[t, j, k]] *= factor;
                        fields.ey.data[[t, j, k]] *= factor;
                        fields.ez.data[[t, j, k]] *= factor;

                        // Right boundary
                        let i = shape.0 - 1 - t;
                        fields.ex.data[[i, j, k]] *= factor;
                        fields.ey.data[[i, j, k]] *= factor;
                        fields.ez.data[[i, j, k]] *= factor;
                    }
                }
            }
        }
    }

    /// Apply PML absorption for H-field (simplified)
    fn apply_pml_h(&self, _fields: &mut ElectromagneticFields) {
        // Similar to apply_pml_e
    }

    /// Time-step the simulation
    pub fn step(&self, fields: &mut ElectromagneticFields) {
        self.update_h_field_parallel(fields);
        self.update_e_field_parallel(fields);
    }

    /// Add a source to the fields
    pub fn add_source(
        &self,
        fields: &mut ElectromagneticFields,
        source: &MaxwellSource,
        time: f64,
    ) {
        use crate::domain::maxwell::MaxwellSourceType;

        match &source.source_type {
            MaxwellSourceType::PointDipole {
                position,
                orientation,
                amplitude,
            } => {
                let signal = amplitude * (2.0 * PI * source.frequency * time).sin();

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

/// Run parallel Maxwell FDTD simulation
pub fn simulate_maxwell_parallel(
    domain: ParallelMaxwellDomain,
    sources: Vec<MaxwellSource>,
    num_steps: usize,
    save_interval: Option<usize>,
) -> Vec<ElectromagneticFields> {
    let mut fields = domain.init_fields();
    let mut results = Vec::new();

    println!(
        "Running parallel FDTD with {} subdomains...",
        domain.decomposition.num_subdomains()
    );

    for step in 0..num_steps {
        let time = step as f64 * domain.dt;

        // Add sources
        for source in &sources {
            domain.add_source(&mut fields, source, time);
        }

        // Update fields in parallel
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
    fn test_parallel_maxwell_creation() {
        let shape = (50, 50, 50);
        let permittivity = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));
        let permeability = WaveArray::from_scalar(shape, Complex::new(1.0, 0.0));

        let domain = ParallelMaxwellDomain::new(
            permittivity,
            permeability,
            1e-6,
            1e-6,
            1e-6,
            [false, false, false],
            [5, 5, 5],
            (2, 2, 2),
        );

        assert_eq!(domain.decomposition.num_subdomains(), 8);
    }
}
