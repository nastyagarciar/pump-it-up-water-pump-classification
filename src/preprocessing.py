"""Reusable preprocessing utilities for the Pump It Up portfolio project.

This is a portfolio refactor of the preprocessing logic used in the original
academic notebook. It keeps train/test transformations consistent by learning
high-cardinality category mappings on the training data only.

Competition datasets are intentionally not included in this repository.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


MISSING_TOKENS = {
    "", " ", "-", "none", "None", "nan", "NaN",
    "unknown", "Unknown", "not known", "Not known",
    "no data", "No data",
}

MASKED_ZERO_COLUMNS = [
    "gps_height",
    "longitude",
    "population",
    "construction_year",
]

DROP_COLUMNS = [
    "recorded_by",
    "id",
    "date_recorded",
    "extraction_type_group",
    "extraction_type_class",
    "payment_type",
    "waterpoint_type_group",
]


class PumpPreprocessor:
    """Fit/transform preprocessing for the water-pump classification task."""

    def __init__(self, top_n_categories: int = 100):
        self.top_n_categories = top_n_categories
        self.frequent_categories_: dict[str, set[str]] = {}

    def fit(self, df: pd.DataFrame) -> "PumpPreprocessor":
        data = df.copy()

        for col in ["funder", "installer"]:
            if col in data.columns:
                values = (
                    data[col]
                    .fillna("missing_category")
                    .astype(str)
                    .str.strip()
                )
                top = values.value_counts().nlargest(self.top_n_categories).index
                self.frequent_categories_[col] = set(top.astype(str))

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        for col in ["public_meeting", "permit"]:
            if col in data.columns:
                data[col] = data[col].astype(str)

        if "date_recorded" in data.columns:
            data["date_recorded"] = pd.to_datetime(
                data["date_recorded"], errors="coerce"
            )
            data["recorded_year"] = data["date_recorded"].dt.year
            data["recorded_month"] = data["date_recorded"].dt.month

        for col in MASKED_ZERO_COLUMNS:
            if col in data.columns:
                data[f"{col}_was_zero"] = (data[col] == 0).astype(int)
                data[col] = data[col].replace(0, np.nan)

        if "construction_year" in data.columns and "recorded_year" in data.columns:
            data["pump_age"] = data["recorded_year"] - data["construction_year"]
            data.loc[data["pump_age"] < 0, "pump_age"] = np.nan

        if "region" in data.columns and "region_code" in data.columns:
            data["region_combined"] = (
                data["region"].astype(str)
                + "_"
                + data["region_code"].astype(str)
            )

        for col, allowed in self.frequent_categories_.items():
            if col in data.columns:
                values = (
                    data[col]
                    .fillna("missing_category")
                    .astype(str)
                    .str.strip()
                )
                data[col] = values.where(values.isin(allowed), "others")

        for col in data.select_dtypes(include=["object"]).columns:
            data[col] = data[col].astype(str).str.strip()
            data[col] = data[col].replace(list(MISSING_TOKENS), np.nan)
            data[col] = data[col].fillna("missing_category")

        data.drop(
            columns=[c for c in DROP_COLUMNS if c in data.columns],
            inplace=True,
        )

        return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
