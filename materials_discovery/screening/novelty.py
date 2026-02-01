"""screening/novelty.py — Check whether a composition is already known."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from pymatgen.core import Composition

logger = logging.getLogger(__name__)


@dataclass
class NoveltyResult:
    """Result of a novelty check against materials databases."""

    composition: dict[str, float]
    is_novel: bool
    closest_known_formula: str | None = None
    closest_distance: float | None = None
    found_in: str | None = None


def _composition_distance(a: Composition, b: Composition) -> float:
    """Fractional L2 distance between two normalised compositions."""
    all_els = sorted(set(a.as_dict()) | set(b.as_dict()))
    va = np.array([a.get_atomic_fraction(el) for el in all_els])
    vb = np.array([b.get_atomic_fraction(el) for el in all_els])
    return float(np.linalg.norm(va - vb))


class KnownCompoundsCache:
    """Pre-fetched cache of all known compounds in a chemical system.

    Fetches from Materials Project (and optionally OQMD) once at init,
    then all novelty checks are pure local comparisons — no API calls.
    """

    def __init__(self, elements: list[str], api_key: str) -> None:
        self.elements = sorted(elements)
        self._known: list[tuple[str, Composition, str]] = []  # (formula, comp, source)
        self._fetch_mp(api_key)
        self._fetch_oqmd()
        logger.info("Novelty cache: %d known compounds loaded", len(self._known))

    def _fetch_mp(self, api_key: str) -> None:
        """Fetch all compounds in the chemical system and all sub-systems from MP."""
        from itertools import combinations
        from mp_api.client import MPRester

        # Query every sub-system (binaries, ternaries, ..., full system).
        subsystems: list[str] = []
        for r in range(2, len(self.elements) + 1):
            for combo in combinations(self.elements, r):
                subsystems.append("-".join(sorted(combo)))

        logger.info("Fetching Materials Project entries for %d sub-systems ...", len(subsystems))
        total = 0
        try:
            with MPRester(api_key) as mpr:
                for chemsys in subsystems:
                    docs = mpr.materials.summary.search(
                        chemsys=chemsys,
                        fields=["formula_pretty", "composition"],
                    )
                    for doc in docs:
                        comp = Composition(doc.composition)
                        self._known.append((doc.formula_pretty, comp, "materials_project"))
                    total += len(docs)
        except Exception as exc:
            logger.warning("MP fetch failed: %s", exc)
        logger.info("  MP: %d entries across %d sub-systems", total, len(subsystems))

    def _fetch_oqmd(self) -> None:
        """Fetch OQMD entries for each element pair/triple (OQMD has no chemsys query)."""
        import qmpy_rester as qr
        from itertools import combinations

        logger.info("Fetching OQMD entries ...")
        count = 0
        # Query for each binary and ternary sub-system.
        for r in range(2, min(len(self.elements) + 1, 4)):
            for combo in combinations(self.elements, r):
                formula_query = "-".join(combo)
                try:
                    with qr.QMPYRester() as q:
                        data = q.get_oqmd_phases(
                            composition=formula_query,
                            verbose=False,
                        )
                    if data and "data" in data:
                        for entry in data["data"]:
                            name = entry.get("name", "")
                            if name:
                                try:
                                    comp = Composition(name)
                                    self._known.append((name, comp, "oqmd"))
                                    count += 1
                                except Exception:
                                    pass
                except Exception:
                    pass
        logger.info("  OQMD: %d entries", count)

    def check(
        self, composition: dict[str, float], tolerance: float = 0.05
    ) -> NoveltyResult:
        """Check a composition against the cached known compounds."""
        target = Composition(composition)

        best_dist = float("inf")
        best_formula = None
        best_source = None

        for formula, known_comp, source in self._known:
            dist = _composition_distance(target, known_comp)
            if dist < best_dist:
                best_dist = dist
                best_formula = formula
                best_source = source

        is_novel = best_dist > tolerance

        if not is_novel:
            logger.debug(
                "KNOWN: %s  closest=%s (dist=%.4f) in %s",
                target.reduced_formula, best_formula, best_dist, best_source,
            )

        return NoveltyResult(
            composition=composition,
            is_novel=is_novel,
            closest_known_formula=best_formula,
            closest_distance=best_dist if best_formula else None,
            found_in=best_source if not is_novel else None,
        )
