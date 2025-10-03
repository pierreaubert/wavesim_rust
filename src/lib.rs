//! WaveSim - A Rust library for simulating wave propagation using the Modified Born Series approach
//!
//! This library provides tools for solving the Helmholtz equation and time-harmonic Maxwell's equations
//! in complex, inhomogeneous media using domain decomposition techniques.

pub mod domain;
pub mod domain_decomposition;
pub mod engine;
pub mod parallel_utils;
pub mod utilities;

// Re-export commonly used types
pub use domain::helmholtz::HelmholtzDomain;
pub use domain::simulation::{simulate, SimulationParams, SimulationResult, Source};
pub use engine::array::{Complex64, WaveArray};

pub mod prelude {
    //! Common imports for using the WaveSim library
    pub use crate::domain::helmholtz::HelmholtzDomain;
    pub use crate::domain::simulation::{simulate, SimulationParams, SimulationResult, Source};
    pub use crate::engine::array::{Complex64, WaveArray};
    pub use crate::utilities::analytical::{
        BoundaryCondition, CircleParams, CircularSolution, RectangleParams, RectangularSolution,
        SphereParams, SphericalSolution,
    };
    pub use crate::utilities::{add_absorbing_boundaries, create_source};
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
