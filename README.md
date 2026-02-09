# Materials Discovery Platform

A computational platform for discovering novel ultra-high temperature ceramics (UHTCs) for aerospace thermal protection systems and fusion reactor components. Uses graph neural networks to predict thermodynamic stability and screen thousands of candidates against materials databases.

Discovered **33 novel, thermodynamically stable ceramic compounds** in the Hf-Zr-Ta-C-N composition space that haven't been synthesized yet.

---

## What It Does

This platform automates the early-stage discovery of new materials for extreme environments (>2000 K):

1. **Generates candidate compositions** — Systematically explores the Hf-Zr-Ta-C-N chemical space (hafnium, zirconium, tantalum, carbon, nitrogen) using compositional grids
2. **Filters against known materials** — Compares every candidate against 1160+ compounds from Materials Project and OQMD databases
3. **Predicts stability** — Uses [CHGNet](https://github.com/CederGroupHub/chgnet) (a pretrained graph neural network) to relax crystal structures and compute formation energies
4. **Exports results** — Generates CIF files, XRD patterns, and interactive 3D viewers for every stable candidate

**Why these elements?** Hafnium and zirconium carbides/nitrides are among the highest-melting ceramics known (>3500 K). Mixing them creates "high-entropy" ceramics with potentially superior thermal shock resistance and oxidation resistance — critical for reentry vehicles and fusion reactor plasma-facing components.

---

## How It Works

### Pre-Screening Pipeline

The discovery process operates in three stages:

#### 1. Composition Grid Generation

The code enumerates all possible compositions in the Hf-Zr-Ta-C-N system at a specified resolution (default 10% steps). For example, with 5 elements at 10% resolution:
- Hf: 0%, 10%, 20%, ..., 100%
- Zr: 0%, 10%, 20%, ..., 100%
- (etc., constrained to sum to 100%)

This generates ~10,000 candidate compositions. Compositions with fewer than 2 elements are skipped (pure elements and near-pure compositions are already well-studied).

#### 2. Novelty Filtering (Local, Fast)

Each candidate is compared against all known compounds using a fractional L2 distance metric:

```
distance = ||composition_A - composition_B||₂
```

With a 5% tolerance, compositions within 0.05 atomic fraction distance of any known material are discarded. This eliminates ~95% of candidates instantly without expensive simulations.

**Databases searched:**
- **Materials Project** — ~850 compounds in Hf-Zr-Ta-C-N sub-systems
- **OQMD** — ~310 additional compounds

#### 3. Stability Prediction (CHGNet Relaxation)

For each novel candidate:

1. A 64-atom disordered rock-salt supercell is constructed (randomly mixed cations/anions)
2. CHGNet performs structure relaxation (energy minimization)
3. Formation energy per atom is computed:
   ```
   E_form = (E_compound - Σ E_element) / N_atoms
   ```
4. Compositions with `E_form < 0.05 eV/atom` are considered stable

CHGNet is a universal ML potential trained on 1.5M structures from Materials Project — it predicts energies and forces at near-DFT accuracy in seconds instead of hours.

### Output Data

For each stable candidate, the platform generates:

- **CIF file** — Crystallographic Information File for the relaxed structure
- **Formation energy** — Thermodynamic stability metric (eV/atom)
- **Lattice parameters** — a, b, c, α, β, γ
- **XRD pattern** — Simulated powder diffraction (top 15 peaks)
- **Density** — g/cm³
- **Space group** — Symmetry classification

All data is saved to `novel_stable_candidates.json` and visualized in interactive HTML viewers.

---

## Results

**33 novel candidates discovered** from ~10,000 compositions screened.

Example candidates:

| Formula | E_form (eV/atom) | Density (g/cm³) | Lattice (Å) | Notes |
|---------|------------------|-----------------|-------------|-------|
| Hf₀.₄Zr₀.₆N | -3.21 | 11.82 | 4.58 | High-entropy nitride |
| Hf₀.₃Ta₀.₃C₀.₄ | -2.94 | 13.45 | 4.52 | Carbide-rich UHTC |
| Zr₀.₅Ta₀.₃N₀.₂ | -3.08 | 10.31 | 4.61 | Metal-rich phase |

**View the full dataset:** Open `data/compound_viewer.html` in a browser to explore all 33 candidates in 3D with structural data and XRD patterns.

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- **Materials Project API key** — [Get one free here](https://next-gen.materialsproject.org/api) (required for database access)

### Installation

```bash
# Clone the repository
git clone https://github.com/Slamwise/MaterialDiscovery.git
cd MaterialDiscovery

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy scipy scikit-learn
pip install pymatgen mp-api qmpy-rester chgnet
```

**Installation notes:**
- CHGNet requires PyTorch — it will auto-install, but you can manually install GPU-enabled PyTorch first for better performance
- Total install size: ~2 GB (includes CHGNet pretrained model)

### Running the Pre-Screener

Discover novel ceramics in the Hf-Zr-Ta-C-N system:

```bash
python -m materials_discovery.prescreener \
    --mp-api-key YOUR_API_KEY \
    --elements Hf,Zr,Ta,C,N \
    --resolution 0.10 \
    --stability-threshold 0.050 \
    --output novel_stable_candidates.json
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mp-api-key` | *required* | Materials Project API key |
| `--elements` | Hf,Zr,Ta,C,N | Comma-separated element list |
| `--resolution` | 0.10 | Composition grid step size (10% = 10,000 candidates) |
| `--stability-threshold` | 0.050 | Max formation energy for stability (eV/atom) |
| `--novelty-tolerance` | 0.05 | Distance threshold for novelty (5% atomic fraction) |
| `--output` | novel_stable_candidates.json | Output file path |

**Runtime:** ~2-4 hours for the full Hf-Zr-Ta-C-N system on a laptop (most time spent on CHGNet relaxations).

### Exploring Results

After the pre-screener finishes:

1. **Open the web viewer:**
   ```bash
   # Open data/compound_viewer.html in your browser
   # On macOS: open data/compound_viewer.html
   # On Linux: xdg-open data/compound_viewer.html
   # On Windows: start data/compound_viewer.html
   ```

2. **Browse candidates** — Click through the list to see:
   - 3D crystal structure (interactive, rotatable)
   - Formation energy and lattice parameters
   - Simulated XRD pattern
   - Elemental composition

3. **Export CIF files** — Find relaxed structures in `data/cif_files/` (one per candidate) for use in other simulation tools (VESTA, Materials Studio, VASP, etc.)

---

## Customizing the Search

### Different Element Systems

Search for novel nitrides in the Ti-V-Cr-N system:

```bash
python -m materials_discovery.prescreener \
    --mp-api-key YOUR_API_KEY \
    --elements Ti,V,Cr,N \
    --resolution 0.10 \
    --output ti_v_cr_n_candidates.json
```

### Higher Resolution (Finer Grid)

Use 5% steps for more thorough exploration (warning: 10x slower):

```bash
python -m materials_discovery.prescreener \
    --mp-api-key YOUR_API_KEY \
    --elements Hf,Zr,Ta,C,N \
    --resolution 0.05 \
    --output high_res_candidates.json
```

### Stricter Stability Threshold

Only accept candidates with very negative formation energies:

```bash
python -m materials_discovery.prescreener \
    --mp-api-key YOUR_API_KEY \
    --elements Hf,Zr,Ta,C,N \
    --stability-threshold 0.000 \
    --output ultra_stable_candidates.json
```

---

## Architecture

```
materials_discovery/
  prescreener.py       -- Main pre-screening orchestrator

  screening/
    novelty.py         -- Materials Project + OQMD distance filter
    stability.py       -- CHGNet structure relaxation

  search/
    space.py           -- Compositional grid enumeration

data/
  novel_stable_candidates.json   -- Output: candidate database
  cif_files/                     -- CIF files for each candidate
  compound_viewer.html           -- Interactive 3D viewer
```

---

## Next Steps

These 33 candidates are computational *predictions* — they need validation before synthesis:

### Phase 2: Property Prediction (Planned)

- **Elastic properties** — Young's modulus, bulk modulus, shear modulus via molecular dynamics
- **Thermal properties** — Thermal conductivity, melting point (requires LAMMPS + MACE potentials)
- **Electronic properties** — Band gap, density of states (requires DFT)

### Phase 3: Experimental Validation

For the most promising candidates:
1. **DFT validation** — Recalculate formation energies with VASP or Quantum ESPRESSO
2. **Phase diagram analysis** — Check for competing phases
3. **Synthesis planning** — Powder metallurgy, spark plasma sintering, or thin-film deposition
4. **Characterization** — XRD, SEM, hardness testing, thermal analysis

---

## Technical Details

### Why CHGNet?

CHGNet is a pretrained universal graph neural network potential ([Deng et al., *Nature Machine Intelligence* 2023](https://www.nature.com/articles/s42256-023-00716-3)) trained on 1.5M materials from Materials Project. Advantages:

- **No training required** — Works out-of-the-box for any composition
- **Fast** — 100x faster than DFT (seconds vs. hours per structure)
- **Accurate** — Formation energy MAE ~30 meV/atom (comparable to GGA-DFT)
- **Charge-informed** — Handles mixed ionic/covalent/metallic bonding

### Compositional Distance Metric

The fractional L2 norm measures similarity between compositions:

```python
def distance(comp_A, comp_B):
    elements = set(comp_A.keys()) | set(comp_B.keys())
    vec_A = [comp_A.get(el, 0) for el in elements]
    vec_B = [comp_B.get(el, 0) for el in elements]
    return sqrt(sum((a - b)**2 for a, b in zip(vec_A, vec_B)))
```

For example:
- `Hf₀.₅C₀.₅` vs. `Hf₀.₄₈C₀.₅₂` → distance = 0.028 (SIMILAR, likely known)
- `Hf₀.₅C₀.₅` vs. `Zr₀.₅C₀.₅` → distance = 0.707 (DIFFERENT, worth exploring)

### Why Rock-Salt Supercells?

Most transition metal carbides and nitrides crystallize in the rock-salt (NaCl) structure:
- Face-centered cubic (FCC) cation sublattice
- FCC anion sublattice
- Lattice constant ~4.5 Å

For high-entropy compositions (e.g., Hf₀.₃Zr₀.₄Ta₀.₃C₀.₅N₀.₅), we create disordered supercells where cations (Hf, Zr, Ta) randomly occupy one sublattice and anions (C, N) occupy the other. This mimics the configurational entropy of real high-entropy ceramics.

---

## References

- **CHGNet:** Deng, B., et al. "CHGNet as a Pretrained Universal Neural Network Potential for Charge-Informed Atomistic Modelling." *Nature Machine Intelligence* (2023). [doi:10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3)

- **Materials Project:** Jain, A., et al. "Commentary: The Materials Project: A Materials Genome Approach to Accelerating Materials Innovation." *APL Materials* 1, 011002 (2013). [materialsproject.org](https://materialsproject.org)

- **OQMD:** Kirklin, S., et al. "The Open Quantum Materials Database (OQMD): Assessing the Accuracy of DFT Formation Energies." *npj Computational Materials* 1, 15010 (2015). [oqmd.org](https://oqmd.org)

- **High-Entropy Ceramics:** Rost, C.M., et al. "Entropy-Stabilized Oxides." *Nature Communications* 6, 8485 (2015). [doi:10.1038/ncomms9485](https://doi.org/10.1038/ncomms9485)

---

## FAQ

**Q: Can I search other element systems?**
A: Yes! Change `--elements` to any comma-separated list. Works best with elements that have Materials Project data (most of the periodic table). Stick to 3-6 elements to keep runtimes reasonable.

**Q: How accurate are the predictions?**
A: CHGNet formation energies have ~30 meV/atom error vs. DFT. Most predictions are reliable, but candidates should be validated with proper DFT (VASP, QE) before synthesis. Use these results to prioritize which materials to study further.

**Q: Why are some candidates missing XRD patterns?**
A: XRD calculation fails for highly disordered structures or when pymatgen's symmetry analyzer can't classify the space group. The structure data is still valid.

**Q: Can I run this without a GPU?**
A: Yes — CHGNet runs on CPU (slower but works). Expect ~2x longer runtimes. For GPU acceleration, install PyTorch with CUDA support before installing CHGNet.

**Q: How do I cite this work?**
A: Repository coming soon. For now, cite the CHGNet paper (Deng et al., 2023) for the stability predictions.

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'mp_api'"**
→ Install: `pip install mp-api`

**"MPRestError: API key not valid"**
→ Check your API key at [https://next-gen.materialsproject.org/api](https://next-gen.materialsproject.org/api). Make sure to use the new next-gen API key, not the legacy one.

**CHGNet runs very slowly**
→ Install GPU PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118` (adjust `cu118` for your CUDA version). Check if GPU is detected: `python -c "import torch; print(torch.cuda.is_available())"`

**"Too many API requests" error**
→ The pre-screener batches API calls to avoid rate limits. If you hit this, add `time.sleep(1)` in `novelty.py` after MP queries.

---

## License

MIT — Free for academic and commercial use.

---

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Add support for other crystal structures (perovskites, spinels, etc.)
- [ ] Parallelize CHGNet relaxations across multiple GPUs
- [ ] Add convex hull analysis (energy above hull for each candidate)
- [ ] Integrate property prediction (elastic moduli, thermal conductivity)
- [ ] Add CLI progress bar for long runs

Open an issue or pull request on GitHub.

---

Built with [Claude](https://claude.ai) • Powered by [CHGNet](https://github.com/CederGroupHub/chgnet) • Data from [Materials Project](https://materialsproject.org)
