"""Configuration dataclasses for the Prometheus discovery pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SearchConfig:
    """Defines the compositional search space."""

    elements: tuple[str, ...] = ("Hf", "Zr", "Ta", "C", "N")
    # Resolution of the compositional grid (0.05 → 5 % increments).
    composition_resolution: float = 0.05
    # Target temperatures for property evaluation (Kelvin).
    temperature_targets_K: tuple[float, ...] = (300.0, 2200.0, 3000.0)


@dataclass(frozen=True)
class BOConfig:
    """Bayesian-optimisation hyper-parameters."""

    surrogate: str = "gp"  # "gp" | "rf"
    acquisition: str = "ucb"
    ucb_kappa: float = 2.576  # ≈ 99 % CI exploration width
    n_initial_random: int = 20
    batch_size: int = 5


@dataclass(frozen=True)
class SimulationConfig:
    """LAMMPS simulation parameters."""

    n_atoms: int = 256
    lattice_constant_A: float = 4.5  # Å — approximate for UHTC rock-salt
    equilibration_steps: int = 10_000
    production_steps: int = 20_000
    timestep_ps: float = 0.001  # 1 fs
    strain_magnitude: float = 0.003
    potential_type: str = "mace"  # "mace" | "nequip"
    potential_path: str = "models/mace_uhtc_v1.model"
    # GPU acceleration — maps to LAMMPS `package gpu` or KOKKOS settings.
    gpu_enabled: bool = True
    gpu_device_id: int = 0


@dataclass(frozen=True)
class FitnessWeights:
    """Weights for multi-objective scoring."""

    elastic_modulus: float = 0.4
    thermal_conductivity: float = 0.3
    cohesive_energy: float = 0.3


@dataclass(frozen=True)
class UncertaintyConfig:
    """ML-IAP committee disagreement settings."""

    threshold_eV_per_atom: float = 0.05
    ensemble_size: int = 4


@dataclass
class EpochConfig:
    """Top-level knobs for a single discovery epoch."""

    search: SearchConfig = field(default_factory=SearchConfig)
    bo: BOConfig = field(default_factory=BOConfig)
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    weights: FitnessWeights = field(default_factory=FitnessWeights)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    n_iterations: int = 50
    mission: str = "aerospace"  # "aerospace" | "fusion"
