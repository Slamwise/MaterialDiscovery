"""store/results.py — Lightweight results ledger backed by SQLite."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch       INTEGER NOT NULL,
    iteration   INTEGER NOT NULL,
    composition TEXT    NOT NULL,
    temperature REAL    NOT NULL,
    E_GPa       REAL,
    kappa_WmK   REAL,
    E_coh_eV    REAL,
    fitness     REAL,
    metadata    TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class ResultsDB:
    def __init__(self, db_path: str | Path = "results.sqlite") -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        logger.info("Results DB opened at %s", db_path)

    def insert(
        self,
        epoch: int,
        iteration: int,
        composition: dict[str, float],
        temperature: float,
        E: float,
        kappa: float,
        E_coh: float,
        fitness: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO results "
            "(epoch, iteration, composition, temperature, E_GPa, kappa_WmK, E_coh_eV, fitness, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                epoch,
                iteration,
                json.dumps(composition),
                temperature,
                E,
                kappa,
                E_coh,
                fitness,
                json.dumps(metadata) if metadata else None,
            ),
        )
        self._conn.commit()

    def top_n(self, n: int = 10) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT composition, temperature, E_GPa, kappa_WmK, E_coh_eV, fitness "
            "FROM results ORDER BY fitness DESC LIMIT ?",
            (n,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()
