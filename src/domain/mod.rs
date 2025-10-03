//! Domain module for wave equation solvers

pub mod domain_trait;
pub mod helmholtz;
pub mod helmholtz_parallel;
pub mod helmholtz_schwarz;
pub mod iteration;
pub mod maxwell;
pub mod maxwell_parallel;
pub mod simulation;

pub use domain_trait::Domain;
pub use helmholtz::HelmholtzDomain;
pub use simulation::{simulate, SimulationParams};
