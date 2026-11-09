"""Generic adapter utilities for future datasets."""
from __future__ import annotations
import pandas as pd


def validate_interactions(frame: pd.DataFrame, user_column: str, item_column: str) -> pd.DataFrame:
    """Drops invalid interaction identifiers for generic dataset adapters."""
    if user_column not in frame or item_column not in frame:
        raise ValueError(f"Interactions require {user_column!r} and {item_column!r} columns.")
    return frame.dropna(subset=[user_column, item_column]).copy()
