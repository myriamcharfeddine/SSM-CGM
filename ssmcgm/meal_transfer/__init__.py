"""Passive meal-state transfer pipeline (CGMacros teacher -> AI-READI weak labels
-> causal student -> structured meal-state decoder).

Phases A-G are documented in ``outputs/no_log_scenarios/meal_transfer/`` and the
brief. Entry point: ``ssmcgm.meal_transfer.pipeline.run_pipeline`` /
``scripts/run_meal_transfer.py``.
"""

from . import config  # noqa: F401

__all__ = ["config"]
