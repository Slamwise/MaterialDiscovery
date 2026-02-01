"""
prescreener.py — Pre-screen candidate compositions for novelty and stability.

Usage
-----
    python -m materials_discovery.prescreener \
        --mp-api-key YOUR_KEY \
        --elements Hf,Zr,Ta,C,N \
        --resolution 0.10 \
        --stability-threshold 0.050

Pipeline:
    1. Fetch all known compounds once (Materials Project + OQMD).
    2. Generate compositional grid.
    3. Filter out already-known compositions (local comparison — fast).
    4. Predict stability of novel candidates via CHGNet.
    5. Output passing candidates to JSON (saved after every hit).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .screening.novelty import KnownCompoundsCache
from .screening.stability import predict_stability
from .search.space import build_composition_grid

logger = logging.getLogger("prescreener")


def _build_rich_entry(
    comp: dict[str, float], stability, candidate_id: int
) -> dict:
    """Extract rich structural data from a StabilityResult for the viewer."""
    from pymatgen.core import Composition
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    entry: dict = {
        "id": candidate_id,
        "composition": comp,
        "formula": Composition(comp).reduced_formula,
        "formation_energy_eV_per_atom": float(stability.formation_energy_eV_per_atom),
        "e_above_hull_eV_per_atom": (
            float(stability.e_above_hull_eV_per_atom)
            if stability.e_above_hull_eV_per_atom is not None
            else None
        ),
        "relaxed_lattice_constant_A": float(stability.relaxed_lattice_constant_A),
    }

    struct = getattr(stability, "relaxed_structure", None)
    if struct is None:
        return entry

    # Lattice parameters
    lat = struct.lattice
    entry["lattice"] = {
        "a": round(float(lat.a), 4),
        "b": round(float(lat.b), 4),
        "c": round(float(lat.c), 4),
        "alpha": round(float(lat.alpha), 2),
        "beta": round(float(lat.beta), 2),
        "gamma": round(float(lat.gamma), 2),
        "volume": round(float(lat.volume), 2),
    }

    # Density
    entry["density_g_per_cm3"] = round(float(struct.density), 4)

    # Number of sites
    entry["n_sites"] = len(struct)

    # Space group (use loose tolerance for disordered supercells)
    try:
        sga = SpacegroupAnalyzer(struct, symprec=0.5)
        entry["spacegroup"] = {
            "symbol": sga.get_space_group_symbol(),
            "number": sga.get_space_group_number(),
            "crystal_system": sga.get_crystal_system(),
        }
    except Exception:
        entry["spacegroup"] = None

    # XRD peaks (top 10 strongest)
    try:
        from pymatgen.analysis.diffraction.xrd import XRDCalculator

        xrd = XRDCalculator()
        pattern = xrd.get_pattern(struct)
        # Get top 10 peaks by intensity
        peaks = sorted(
            zip(pattern.x, pattern.y, pattern.hkls),
            key=lambda p: p[1],
            reverse=True,
        )[:15]
        entry["xrd_peaks"] = [
            {
                "two_theta": round(float(p[0]), 3),
                "intensity": round(float(p[1]), 2),
                "hkl": str(p[2][0]["hkl"]) if p[2] else "",
            }
            for p in peaks
        ]
    except Exception as exc:
        logger.debug("XRD failed: %s", exc)
        entry["xrd_peaks"] = None

    # Atomic positions (for 3D viewer — use a primitive cell or just first 32 sites)
    sites = []
    for site in struct[:min(len(struct), 64)]:
        sites.append({
            "element": str(site.specie),
            "x": round(float(site.coords[0]), 4),
            "y": round(float(site.coords[1]), 4),
            "z": round(float(site.coords[2]), 4),
            "frac": [round(float(f), 4) for f in site.frac_coords],
        })
    entry["sites"] = sites

    # Lattice matrix for the 3D viewer
    entry["lattice_matrix"] = [
        [round(float(x), 4) for x in row] for row in lat.matrix.tolist()
    ]

    return entry


def run_prescreening(
    elements: list[str],
    resolution: float,
    api_key: str,
    stability_threshold: float = 0.050,
    novelty_tolerance: float = 0.05,
    skip_novelty: bool = False,
    output_path: str = "novel_stable_candidates.json",
) -> list[dict]:
    """Run the full novelty + stability screening pipeline."""

    # ── Pre-fetch all known compounds (one-time cost) ──
    cache = None
    if not skip_novelty:
        logger.info("Building known-compounds cache (one-time fetch) ...")
        cache = KnownCompoundsCache(elements, api_key)

    grid = build_composition_grid(n_elements=len(elements), resolution=resolution)
    logger.info("Composition grid: %d candidates  elements=%s", len(grid), elements)

    candidates: list[dict] = []
    n_known = 0
    n_unstable = 0
    n_novel_stable = 0
    n_skipped = 0

    for i, vec in enumerate(grid):
        comp = {el: round(float(vec[j]), 4) for j, el in enumerate(elements)}

        # Skip near-pure compositions (fewer than 2 nonzero elements).
        nonzero = sum(1 for v in comp.values() if v > 0.01)
        if nonzero < 2:
            n_skipped += 1
            continue

        comp_str = " ".join(f"{k}{v:.0%}" for k, v in comp.items() if v > 0.01)

        # Step 1: Novelty check (local — instant).
        if cache is not None:
            novelty = cache.check(comp, tolerance=novelty_tolerance)
            if not novelty.is_novel:
                n_known += 1
                continue

        # Step 2: Stability prediction via CHGNet.
        logger.info(
            "[%d/%d] Predicting stability: %s  (novel=%d  known=%d  unstable=%d)",
            i + 1, len(grid), comp_str, n_novel_stable, n_known, n_unstable,
        )
        try:
            stability = predict_stability(
                comp,
                api_key=api_key,
                stability_threshold_eV=stability_threshold,
            )
        except Exception as exc:
            logger.warning("  Stability prediction failed: %s", exc)
            continue

        if not stability.is_stable:
            logger.info(
                "  UNSTABLE — E=%.4f eV/atom  e_above_hull=%s",
                stability.formation_energy_eV_per_atom,
                f"{stability.e_above_hull_eV_per_atom:.4f}"
                if stability.e_above_hull_eV_per_atom is not None
                else "N/A",
            )
            n_unstable += 1
            continue

        # Passed both checks.
        n_novel_stable += 1
        entry = _build_rich_entry(comp, stability, n_novel_stable)
        candidates.append(entry)
        logger.info(
            "  NOVEL + STABLE #%d  E=%.4f eV/atom  hull=%.4f  a=%.3f A",
            n_novel_stable,
            stability.formation_energy_eV_per_atom,
            stability.e_above_hull_eV_per_atom or 0.0,
            stability.relaxed_lattice_constant_A,
        )

        # Save CIF file for the relaxed structure.
        if stability.relaxed_structure is not None:
            cif_dir = os.path.join(os.path.dirname(output_path) or ".", "cif_files")
            os.makedirs(cif_dir, exist_ok=True)
            cif_path = os.path.join(cif_dir, f"candidate_{n_novel_stable:04d}.cif")
            try:
                stability.relaxed_structure.to(filename=cif_path)
            except Exception as exc:
                logger.warning("  CIF export failed: %s", exc)

        # Save incrementally.
        with open(output_path, "w") as f:
            json.dump(candidates, f, indent=2)

    # Final write.
    with open(output_path, "w") as f:
        json.dump(candidates, f, indent=2)

    logger.info("=== PRESCREENING COMPLETE ===")
    logger.info("  Total screened:  %d", n_known + n_unstable + n_novel_stable)
    logger.info("  Skipped (pure):  %d", n_skipped)
    logger.info("  Already known:   %d", n_known)
    logger.info("  Unstable:        %d", n_unstable)
    logger.info("  Novel + stable:  %d", n_novel_stable)
    logger.info("  Output: %s", output_path)

    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="prescreener",
        description="Pre-screen compositions for novelty and thermodynamic stability.",
    )
    parser.add_argument("--mp-api-key", required=True, help="Materials Project API key")
    parser.add_argument(
        "--elements",
        default="Hf,Zr,Ta,C,N",
        help="Comma-separated element list (default: Hf,Zr,Ta,C,N)",
    )
    parser.add_argument("--resolution", type=float, default=0.10)
    parser.add_argument("--stability-threshold", type=float, default=0.050)
    parser.add_argument("--novelty-tolerance", type=float, default=0.05)
    parser.add_argument("--skip-novelty", action="store_true")
    parser.add_argument("--output", default="novel_stable_candidates.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("prescreener.log"),
        ],
    )

    elements = [e.strip() for e in args.elements.split(",")]

    run_prescreening(
        elements=elements,
        resolution=args.resolution,
        api_key=args.mp_api_key,
        stability_threshold=args.stability_threshold,
        novelty_tolerance=args.novelty_tolerance,
        skip_novelty=args.skip_novelty,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
