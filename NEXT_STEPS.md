# Prometheus Phase 1 — Next Steps

## Current Status

Phase 1 prescreening is complete. The pipeline has identified **33 novel, thermodynamically stable ceramic candidates** in the Hf-Zr-Ta-C-N composition space using CHGNet-predicted formation energies and novelty filtering against Materials Project + OQMD (1,160 known compounds).

However, CHGNet stability alone is insufficient to confirm synthesizability. The current filter (`E_form < 0.05 eV/atom`) only checks whether a compound is energetically bound — not whether it is stable against decomposition into competing phases, dynamically stable, or mechanically viable at operating temperatures.

The 33 candidates must now pass through a series of increasingly expensive evaluations before any can be recommended for experimental synthesis.

---

## Evaluation Pipeline

Evaluations are ranked by importance. The first five are **critical** — each one can eliminate candidates and should be run sequentially to avoid wasting compute on compounds that fail earlier filters.

| Rank | Evaluation | Compute (1-10) | Time (33 candidates) | Hardware | Critical? | Rationale |
|---|---|---|---|---|---|---|
| 1 | Energy above convex hull | 1 | Minutes | Laptop CPU | **YES** | Gatekeeping filter — if a candidate decomposes into competing phases, nothing else matters |
| 2 | Phonon stability (Phonopy + DFT) | 7 | 1–2 weeks | HPC cluster | **YES** | Imaginary phonon modes = structure doesn't physically exist, regardless of energy |
| 3 | DFT structural relaxation | 6 | 2–5 days | HPC cluster | **YES** | Ground truth for formation energy — validates/invalidates CHGNet predictions that everything downstream depends on |
| 4 | Elastic constants (MACE/LAMMPS) | 3 | Hours | 1 GPU | **YES** | Core mission metric — a UHTC with poor mechanical properties is useless for aerospace/fusion |
| 5 | Melting point (MD heating ramp) | 5 | 1–2 days | 1 GPU | **YES** | Defines UHTC classification — must survive >2000 K or it's not a candidate |
| 6 | Thermal conductivity (Green-Kubo) | 4 | ~1 day | 1 GPU | No | Important for mission scoring (especially fusion PFC) but doesn't gate synthesizability |
| 7 | MACE/NequIP training | 6 | 2–5 days | 1–4 GPUs | No | Enables the active-learning loop but not required to validate current 33 candidates |
| 8 | Band gap (DFT, PBE) | 5 | 1–2 days | Small HPC cluster | No | Electronic characterization — most UHTCs are metallic, so unlikely to be a surprise |
| 9 | Oxidation resistance (surface DFT) | 8 | 2–4 weeks | Large HPC | No | Matters for real deployment but premature before confirming basic stability |
| 10 | Band gap (HSE06/GW) | 8 | 2–4 weeks | Large HPC | No | Precision refinement — only needed if PBE reveals unexpected semiconducting behavior |

### Compute Scale Reference

- **1–3**: Laptop or single GPU, minutes to hours
- **4–5**: Single GPU, hours to days
- **6–7**: HPC cluster or multi-GPU, days to weeks
- **8+**: Large HPC allocation, weeks of wall time

---

## Sequential Kill Chain

The critical evaluations form a sequential filter. Each stage can eliminate candidates before compute is spent on later stages:

```
Candidate passes convex hull?  ──NO──→  DISCARD
        │ YES
DFT confirms stability?        ──NO──→  DISCARD
        │ YES
Phonons all real?              ──NO──→  DISCARD
        │ YES
Melting point > 2000 K?        ──NO──→  DISCARD
        │ YES
Elastic modulus viable?        ──NO──→  DEPRIORITIZE
        │ YES
        ▼
   SYNTHESIS CANDIDATE
   (proceed to remaining evals)
```

This ordering minimizes wasted HPC hours. The convex hull check is essentially free and may eliminate a significant fraction of the 33 candidates in minutes.

---

## Detailed Reasoning by Evaluation

### 1. Energy Above Convex Hull (Critical)

**Why first:** The current CHGNet filter only asks "is this compound bound?" — it does not ask "is this compound stable against decomposition into simpler competing phases?" A compound with negative formation energy can still be thermodynamically unstable if a mixture of other known phases has even lower energy.

**Method:** Use pymatgen's `PhaseDiagram` and `PDAnalyzer` to compute the energy above the convex hull for each candidate against all known phases in the Hf-Zr-Ta-C-N system.

**Thresholds:**
- `E_above_hull < 50 meV/atom` → ground-state stable, conventional synthesis likely works
- `50–100 meV/atom` → metastable, but may be **entropy-stabilized** at high temperature (relevant for high-entropy ceramics). Calculate: `T_stabilization = ΔH / ΔS_config`
- `> 100 meV/atom` → likely requires non-equilibrium synthesis (thin films, rapid quenching, ball milling)

**Implementation:** This is a code change in the existing pipeline. Pymatgen already provides the necessary tools, and Materials Project API data is already being fetched in `screening/novelty.py`.

### 2. Phonon Stability (Critical)

**Why second:** A compound can sit at a local energy minimum (negative formation energy, below the hull) but still be dynamically unstable — meaning the crystal structure will spontaneously distort. Imaginary (negative) phonon frequencies indicate that the structure is at a saddle point, not a true minimum.

**Method:** Phonopy with DFT force constants. Create supercell displacements, compute forces via VASP/QE, and check the phonon dispersion for imaginary modes.

**Cost note:** This is the most expensive critical evaluation (compute 7/10). Only run it on candidates that survive the convex hull filter and DFT relaxation.

### 3. DFT Structural Relaxation (Critical)

**Why third:** CHGNet is a universal ML potential with ~30 meV/atom accuracy. For novel compositions far from its training distribution, errors could be larger. DFT provides the ground truth formation energy and relaxed structure.

**Method:** Full ionic + cell relaxation in VASP (or QE) with PBE functionals. Compare DFT formation energies against CHGNet predictions. If discrepancies exceed ~50 meV/atom, the CHGNet result should not be trusted.

**Bonus:** DFT-relaxed structures serve as training data for MACE/NequIP fine-tuning (evaluation #7), so this compute is not wasted.

### 4. Elastic Constants (Critical)

**Why fourth:** The entire mission — aerospace TPS and fusion PFC — requires high mechanical strength at extreme temperatures. A compound that is thermodynamically and dynamically stable but mechanically weak is not useful.

**Method:** Already implemented in `fitness/mechanical.py`. Apply 6 independent Voigt strains, compute the stress tensor response, assemble the 6x6 elastic stiffness matrix C_ij, and extract bulk/shear/Young's modulus via Voigt-Reuss-Hill averaging.

**Targets:**
- Bulk modulus > 200 GPa (aerospace), > 150 GPa (fusion)
- Positive definite C_ij matrix (mechanical stability criterion)
- Pugh ratio B/G > 1.75 suggests ductility (desirable for thermal shock resistance)

### 5. Melting Point Estimation (Critical)

**Why fifth:** "Ultra-high temperature ceramic" means operational above 2000 K. If the predicted melting point is below this threshold, the compound fails the fundamental UHTC definition regardless of other properties.

**Method:** MD heating ramp (gradual temperature increase, monitor for discontinuity in volume/energy) or solid-liquid coexistence method (more accurate but more expensive). Use MACE potential if trained, or CHGNet as fallback.

**Caveat:** MD melting points can overestimate by 10–20% due to superheating of the perfect crystal. The coexistence method avoids this but requires larger supercells (~1000+ atoms).

### 6. Thermal Conductivity (Non-critical)

**Method:** Already implemented in `fitness/thermal.py` via Green-Kubo autocorrelation. Important for mission-specific ranking (fusion applications prioritize high thermal conductivity) but does not determine whether a compound can be synthesized.

### 7. MACE/NequIP Training (Non-critical)

**Purpose:** Fine-tune a machine-learned interatomic potential on the DFT data generated in evaluations #2–3. This enables the Bayesian optimization active-learning loop to run with meaningful elastic/thermal predictions for future discovery epochs. Not needed to validate the current 33 candidates.

### 8–10. Band Gap and Oxidation Resistance (Non-critical)

These are characterization-level evaluations that matter for eventual deployment but are premature before confirming basic thermodynamic, dynamic, and mechanical viability.

---

## Synthesizability Assessment Framework

Synthesizability is not a single number — it is a multi-dimensional assessment:

### Tier 1: Thermodynamic Feasibility
- Energy above convex hull (evaluation #1)
- Entropy stabilization temperature for metastable phases
- Configurational entropy: `ΔS_config = -k_B Σ x_i ln(x_i)` for high-entropy compositions

### Tier 2: Dynamic Stability
- Phonon dispersion (evaluation #2)
- No imaginary modes = true energy minimum

### Tier 3: Synthesis Route Feasibility
- **Spark Plasma Sintering (SPS)** — standard for UHTCs, works for most rock-salt carbides/nitrides, achievable temperatures up to ~2500°C
- **Arc melting** — for refractory compositions (Hf/Zr/Ta-rich), can reach >3000°C
- **Reactive hot pressing** — metal powders + C or N₂ gas at elevated temperature and pressure
- Check commercial availability of precursor powders (HfC, ZrC, TaC, TaN, etc.)

### Tier 4: Literature Cross-Check
- Search ICSD and Springer Materials for closely related compositions
- If parent binaries (HfC, TaC, ZrN, etc.) are well-characterized, solid solutions between them are likely synthesizable
- High-entropy ceramics in similar systems (e.g., (Hf,Zr,Ta,Nb,Ti)C) have been experimentally demonstrated

---

## Implementation Roadmap

### Phase 2A: Validation (weeks 1–4)
1. Implement convex hull analysis in the pipeline (days 1–2)
2. Set up DFT environment (VASP/QE + pseudopotentials) (days 3–5)
3. Run DFT relaxation on hull-stable candidates (days 5–14)
4. Run phonon calculations on DFT-confirmed candidates (days 14–28)

### Phase 2B: Property Prediction (weeks 3–6)
5. Run elastic constant calculations on surviving candidates (days 15–16)
6. Run melting point estimations (days 17–20)
7. Run thermal conductivity calculations (days 20–22)
8. Score and rank final candidates by mission fitness

### Phase 2C: Active Learning (weeks 5–8)
9. Train MACE on accumulated DFT dataset (days 28–33)
10. Run Bayesian optimization loop with trained potential (days 33–40)
11. Iterate: new candidates → DFT → retrain → optimize

### Phase 3: Experimental (weeks 8+)
12. Select top 3–5 candidates for synthesis
13. Procure precursor powders
14. Synthesize via SPS or arc melting
15. Characterize: XRD, SEM, nanoindentation, thermal diffusivity

---

## Open Questions

- [ ] HPC access: Do we have an allocation for VASP/QE calculations?
- [ ] VASP license: Is a VASP license available, or should we use Quantum ESPRESSO (open source)?
- [ ] Pseudopotentials: Which PAW datasets should be used for Hf, Zr, Ta, C, N?
- [ ] Experimental collaborator: Is there a lab partner for Phase 3 synthesis?
- [ ] Expanded search space: Should we add additional elements (Ti, Nb, W) in future epochs?
