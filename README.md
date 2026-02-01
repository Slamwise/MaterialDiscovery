# Prometheus Phase 1: Active-Learning Materials Discovery

A high-throughput computational platform for discovering novel ultra-high temperature ceramics (UHTCs) and fusion-grade materials using Bayesian optimization, graph neural network interatomic potentials, and molecular dynamics simulation.

---

## Motivation

Designing materials for extreme environments -- aerospace thermal protection systems, fusion reactor plasma-facing components -- demands ceramics that survive above 2000 K while maintaining mechanical integrity and thermal management. Traditional experimental discovery is slow and expensive. This project replaces trial-and-error with an **autonomous active-learning loop** that explores the Hf-Zr-Ta-C-N composition space, screens for thermodynamic stability, and evaluates mechanical and thermal properties entirely in silico.

## How It Works

The pipeline operates in three stages:

### 1. Pre-Screening: Novelty and Stability

Before any expensive simulation, candidate compositions are filtered:

- **Novelty check** -- Each composition is compared against the Materials Project and OQMD databases (~1160 known compounds in the Hf-Zr-Ta-C-N system). A fractional L2 distance metric with a 5% tolerance ensures we only pursue genuinely new materials.
- **Stability prediction** -- Novel candidates are built as disordered rock-salt supercells and relaxed using [CHGNet](https://github.com/CederGroupHub/chgnet), a universal graph neural network potential. Compositions with negative formation energy per atom pass to the next stage.

### 2. Active Learning via Bayesian Optimization

The core discovery engine is a Bayesian optimization loop:

- **Surrogate model** -- A Gaussian Process regressor (switching to Random Forest above 5,000 observations) learns the mapping from composition to fitness.
- **Acquisition function** -- Upper Confidence Bound (UCB) with kappa=2.576 balances exploration of unknown regions with exploitation of promising compositions.
- **Batch selection** -- A greedy kriging-believer heuristic selects diverse batches of 5 compositions per iteration, avoiding redundant sampling.

Each iteration: suggest compositions, simulate properties, update the surrogate, repeat.

### 3. Property Evaluation via Molecular Dynamics

For each candidate composition, a 256-atom disordered rock-salt supercell is constructed and simulated using LAMMPS with trained MACE interatomic potentials:

- **Elastic modulus** -- Six independent Voigt strains are applied. The full 6x6 elastic tensor is assembled from stress responses, then bulk and shear moduli are extracted via Voigt-Reuss-Hill averaging to yield Young's modulus E.
- **Thermal conductivity** -- After NVT equilibration, an NVE production run collects heat-flux autocorrelation data. The Green-Kubo relation converts this to isotropic thermal conductivity kappa.
- **Cohesive energy** -- Computed as the difference between the relaxed bulk energy and isolated atomic energies.

A multi-objective fitness function combines these properties with mission-specific weights:

| Mission | E (stiffness) | kappa (thermal) | E_coh (bonding) |
|---------|--------------|-----------------|-----------------|
| Aerospace | 0.4 | 0.3 | 0.3 |
| Fusion | 0.3 | 0.4 | 0.3 |

### Uncertainty Monitoring

An ensemble of 4 MACE models acts as a committee. When per-atom energy disagreement exceeds 0.05 eV, the configuration is flagged for DFT validation, closing the loop between ML predictions and first-principles ground truth.

## Architecture

```
materials_discovery/
  config.py          -- Dataclasses for search, simulation, and scoring parameters
  main.py            -- Epoch orchestrator (the active-learning loop)
  prescreener.py     -- Pre-screening pipeline (novelty + CHGNet stability)

  search/
    bayesian_opt.py  -- GP/RF surrogate, UCB acquisition, batch selection
    acquisition.py   -- UCB, Expected Improvement, Probability of Improvement
    space.py         -- Compositional grid enumeration

  simulation/
    engine.py        -- In-memory LAMMPS driver (zero file I/O)
    potentials.py    -- ML-IAP model registry
    structure.py     -- Composition vector utilities

  fitness/
    mechanical.py    -- Voigt-Reuss-Hill elastic modulus
    thermal.py       -- Green-Kubo thermal conductivity
    scorer.py        -- Multi-objective fitness aggregation

  ml_iap/
    uncertainty.py   -- Committee disagreement monitor
    trainer.py       -- Online MACE fine-tuning
    dft_oracle.py    -- DFT submission interface (VASP/CP2K)

  screening/
    novelty.py       -- Materials Project / OQMD distance filter
    stability.py     -- CHGNet structure relaxation

  store/
    results.py       -- SQLite results ledger
    checkpoint.py    -- Loop checkpoint save/load
```

## Replication

### Prerequisites

- Python 3.11+
- LAMMPS compiled with KOKKOS + CUDA support (for GPU acceleration)
- A Materials Project API key ([get one here](https://materialsproject.org/api))
- Trained MACE model file (e.g., `models/mace_uhtc_v1.model`)

### Setup

```bash
git clone https://github.com/Slamwise/MaterialDiscovery.git
cd MaterialDiscovery
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install mace-torch chgnet pymatgen mp-api qmpy-rester
```

### Run Pre-Screening

Enumerate the Hf-Zr-Ta-C-N composition space, filter against known databases, and predict stability with CHGNet:

```bash
python -m materials_discovery.prescreener \
    --mp-api-key YOUR_API_KEY \
    --elements Hf,Zr,Ta,C,N \
    --resolution 0.10 \
    --stability-threshold 0.050 \
    --output novel_stable_candidates.json
```

Output: `novel_stable_candidates.json` containing compositions, predicted formation energies, lattice parameters, and CIF structures in `cif_files/`.

### Run the Active-Learning Loop

```bash
python -m materials_discovery.main \
    --potential models/mace_uhtc_v1.model \
    --mission aerospace \
    --iterations 50 \
    --batch-size 5 \
    --temperature 2200 \
    --gpu-device 0
```

Results are logged to `prometheus_results.sqlite` and `prometheus_epoch.log`. Top candidates are ranked by multi-objective fitness at the end of each epoch.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--elements` | Hf,Zr,Ta,C,N | Element palette |
| `--resolution` | 0.05 | Compositional grid step (5%) |
| `--temperature` | 2200 | Target temperature in Kelvin |
| `--mission` | aerospace | Fitness weights (aerospace or fusion) |
| `--batch-size` | 5 | Compositions per BO iteration |
| `--iterations` | 50 | Number of BO iterations per epoch |
| `--gpu-device` | 0 | CUDA device index |

## Design Decisions

**Zero file I/O** -- The LAMMPS engine operates entirely in-memory via C bindings. No dump files, data files, or intermediate logs are written to disk. This eliminates I/O bottleneck in high-throughput loops and avoids disk-space issues when evaluating thousands of candidates.

**Automatic surrogate switching** -- Gaussian Processes provide excellent uncertainty estimates but scale as O(n^3). Above 5,000 observations the optimizer transparently switches to Random Forest, maintaining throughput without sacrificing too much exploration quality.

**Seeded reproducibility** -- NumPy generators are seeded throughout, velocity initialization is deterministic, and checkpoints enable exact loop resumption after interruption.

## References

- Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields," NeurIPS 2022
- Deng et al., "CHGNet as a Pretrained Universal Neural Network Potential for Charge-Informed Atomistic Modelling," Nature Machine Intelligence 2023
- Hill, R., "The Elastic Behaviour of a Crystalline Aggregate," Proceedings of the Physical Society A 65, 1952
- Green, M.S., "Markoff Random Processes and the Statistical Mechanics of Time-Dependent Phenomena," J. Chem. Phys. 22, 1954

## License

MIT
