"""light IO helpers for this project"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def read_data_pd(path: str | Path, **read_kwargs: Any) -> pd.DataFrame:
    """Read CSV, run df.info(show_counts=True), return the DataFrame."""
    csv_path = Path(path)
    df = pd.read_csv(csv_path, low_memory=False, **read_kwargs)
    df.info(show_counts=True)
    return df
