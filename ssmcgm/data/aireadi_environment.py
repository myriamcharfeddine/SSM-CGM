"""Thin, additive extension of the AI-READI stream schema for environmental exposure.

Does not modify ``ssmcgm/data/aireadi.py``. Calls the existing
``infer_or_validate_schema`` unchanged, then appends the causal environmental
columns produced by ``Preprocessing/prepare_environmental_features.py`` (when
present in the panel dataframe) to ``dynamic_reals`` -- the same list every other
historical stream feature is drawn from. No other schema field is touched, and no
model code needs to change: ``AireadiPreprocessor.fit`` and
``GroupedLinearFusion`` are both generic over ``dynamic_reals``.
"""
from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd

from ssmcgm.data.aireadi import AireadiSchema, infer_or_validate_schema

ENVIRONMENT_DYNAMIC_COLUMNS = [
    "env_temp_mean",
    "env_hum_mean",
    "env_temp_var",
    "env_hum_var",
    "env_age_since_last_min",
    "env_missing",
]


def infer_or_validate_schema_with_environment(
    df: pd.DataFrame, schema: Optional[Mapping[str, object]] = None
) -> AireadiSchema:
    base = infer_or_validate_schema(df, schema)
    env_cols_present = [c for c in ENVIRONMENT_DYNAMIC_COLUMNS if c in df.columns]
    missing = [c for c in ENVIRONMENT_DYNAMIC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Environmental columns missing from panel; run "
            "Preprocessing/prepare_environmental_features.py first. "
            f"Missing: {missing}"
        )
    base.dynamic_reals = list(base.dynamic_reals) + env_cols_present
    return base
