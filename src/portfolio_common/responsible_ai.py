"""Utilities for a simple responsible AI checklist."""

from __future__ import annotations

import pandas as pd


def fairness_check(data: pd.DataFrame, sensitive_column: str, outcome_column: str) -> dict[str, float]:
    """Return a simple rate gap between protected groups and the outcome."""
    rates = data.groupby(sensitive_column)[outcome_column].mean()
    if len(rates) <= 1:
        return {"rate_gap": 0.0}
    return {"rate_gap": float(abs(rates.max() - rates.min()))}
