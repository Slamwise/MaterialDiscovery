"""
main.py — Prometheus Phase 1: First Epoch Orchestrator.

Usage
-----
    python -m materials_discovery.main \
        --potential models/mace_uhtc_v1.model \
        --mission aerospace \
        --iterations 50 \
        --batch-size 5 \
        --temperature 2200 \
        --gpu-device 0

The loop:
    1. Bayesian Optimizer suggests a batch of compositions.
    2. For each candidate, build a 256-atom supercell in LAMMPS (in memory).
    3. NPT equilibration at target temperature.
    4. Measure elastic modulus E (Voigt–Reuss–Hill).
    5. Measure thermal conductivity κ (Green–Kubo).
    6. Score fitness → feed back to the optimizer.
    7. If ML-IAP uncertainty is high → flag for DFT (future: auto-retrain).
    8. Repeat.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

from .config import BOConfig, EpochConfig, FitnessWeights, SearchConfig, SimulationConfig
from .fitness.mechanical import elastic_modulus_vrh
from .fitness.scorer import FitnessResult, aerospace_fitness, fusion_fitness
from .fitness.thermal import thermal_conductivity_green_kubo
from .search.bayesian_opt import BayesianOptimizer
from .simulation.engine import LammpsEngine
from .store.results import ResultsDB

logger = logging.getLogger("prometheus")


def run_epoch(cfg: EpochConfig) -> list[FitnessResult]:
    """Execute one full discovery epoch and return all scored results."""

    elements = list(cfg.search.elements)
    optimizer = BayesianOptimizer(
        elements=elements,
        resolution=cfg.search.composition_resolution,
        kappa=cfg.bo.ucb_kappa,
        n_initial_random=cfg.bo.n_initial_random,
    )
    db = ResultsDB("prometheus_results.sqlite")
    results: list[FitnessResult] = []
    score_fn = aerospace_fitness if cfg.mission == "aerospace" else fusion_fitness
    T = cfg.search.temperature_targets_K[1] if len(cfg.search.temperature_targets_K) > 1 else 2200.0

    logger.info(
        "═══ PROMETHEUS EPOCH START ═══  mission=%s  T=%.0f K  "
        "iterations=%d  batch=%d  grid_res=%.2f",
        cfg.mission, T, cfg.n_iterations, cfg.bo.batch_size,
        cfg.search.composition_resolution,
    )

    for iteration in range(cfg.n_iterations):
        candidates = optimizer.suggest(cfg.bo.batch_size)
        iter_start = time.perf_counter()

        for cand in candidates:
            # Skip degenerate compositions (all weight on one element).
            nonzero = sum(1 for v in cand.composition.values() if v > 0.01)
            if nonzero < 2:
                logger.debug("Skipping near-pure composition: %s", cand.composition)
                continue

            with LammpsEngine(
                potential_path=cfg.sim.potential_path,
                model_type=cfg.sim.potential_type,
                gpu_enabled=cfg.sim.gpu_enabled,
                gpu_device_id=cfg.sim.gpu_device_id,
            ) as engine:
                try:
                    engine.build(
                        composition=cand.composition,
                        n_atoms=cfg.sim.n_atoms,
                        lattice_constant=cfg.sim.lattice_constant_A,
                    )

                    # ── Phase 1: NPT equilibration ──
                    engine.run_npt(
                        T=T,
                        P=0.0,
                        steps=cfg.sim.equilibration_steps,
                        dt=cfg.sim.timestep_ps,
                    )

                    # ── Phase 2: Elastic modulus ──
                    E = elastic_modulus_vrh(engine, strain_mag=cfg.sim.strain_magnitude)

                    # ── Phase 3: NVT thermalisation before Green–Kubo ──
                    engine.run_nvt(T=T, steps=cfg.sim.equilibration_steps // 2, dt=cfg.sim.timestep_ps)

                    # ── Phase 4: Thermal conductivity ──
                    try:
                        kappa = thermal_conductivity_green_kubo(
                            engine,
                            T=T,
                            correlation_length=cfg.sim.production_steps,
                            dt=cfg.sim.timestep_ps,
                        )
                    except Exception as _kappa_err:
                        logger.warning(
                            "Thermal conductivity failed (%s); using NaN fallback", _kappa_err
                        )
                        kappa = float("nan")

                    # ── Phase 5: Cohesive energy ──
                    E_coh = engine.get_pe_per_atom()

                    # ── Score ──
                    fitness = score_fn(E, kappa, E_coh, cfg.weights)

                    result = FitnessResult(
                        composition=cand.composition,
                        temperature_K=T,
                        elastic_modulus_GPa=E,
                        thermal_conductivity_WmK=kappa,
                        cohesive_energy_eV=E_coh,
                        fitness_score=fitness,
                    )
                    results.append(result)
                    optimizer.observe(cand.composition_vec, fitness)

                    db.insert(
                        epoch=0,
                        iteration=iteration,
                        composition=cand.composition,
                        temperature=T,
                        E=E,
                        kappa=kappa,
                        E_coh=E_coh,
                        fitness=fitness,
                    )

                    logger.info(
                        "[iter %3d] %s  →  E=%6.1f GPa  κ=%6.2f W/mK  "
                        "E_coh=%6.3f eV  fitness=%.4f",
                        iteration,
                        {k: round(v, 2) for k, v in cand.composition.items() if v > 0.01},
                        E, kappa, E_coh, fitness,
                    )

                except Exception:
                    logger.exception(
                        "Simulation failed for %s — skipping.", cand.composition
                    )

        elapsed = time.perf_counter() - iter_start
        logger.info(
            "[iter %3d] batch complete  (%.1f s)  observations=%d",
            iteration, elapsed, optimizer.n_observed,
        )

    # ── Summary ──
    results.sort(key=lambda r: r.fitness_score, reverse=True)
    logger.info("═══ EPOCH COMPLETE ═══  total evaluated: %d", len(results))
    logger.info("═══ TOP 5 CANDIDATES ═══")
    for rank, r in enumerate(results[:5], 1):
        logger.info(
            "  #%d  %s  fitness=%.4f  E=%.1f  κ=%.2f  E_coh=%.3f",
            rank,
            {k: round(v, 2) for k, v in r.composition.items() if v > 0.01},
            r.fitness_score, r.elastic_modulus_GPa,
            r.thermal_conductivity_WmK, r.cohesive_energy_eV,
        )

    if results:
        best_comp, best_fit = optimizer.best_so_far()
        logger.info("BEST OVERALL: %s  fitness=%.4f", best_comp, best_fit)

    db.close()
    return results


# ── CLI ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="prometheus",
        description="Prometheus Phase 1 — Active-learning materials discovery",
    )
    parser.add_argument("--potential", required=True, help="Path to ML-IAP model file")
    parser.add_argument("--potential-type", default="mace", choices=["mace", "nequip"])
    parser.add_argument("--mission", default="aerospace", choices=["aerospace", "fusion"])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=2200.0)
    parser.add_argument("--n-atoms", type=int, default=256)
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("prometheus_epoch.log"),
        ],
    )

    cfg = EpochConfig(
        search=SearchConfig(
            temperature_targets_K=(300.0, args.temperature, 3000.0),
            composition_resolution=args.resolution,
        ),
        bo=BOConfig(batch_size=args.batch_size),
        sim=SimulationConfig(
            n_atoms=args.n_atoms,
            potential_path=args.potential,
            potential_type=args.potential_type,
            gpu_enabled=not args.no_gpu,
            gpu_device_id=args.gpu_device,
        ),
        n_iterations=args.iterations,
        mission=args.mission,
    )

    run_epoch(cfg)


if __name__ == "__main__":
    main()
