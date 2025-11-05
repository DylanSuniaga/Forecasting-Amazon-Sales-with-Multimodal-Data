"""utility functions to prep the text data before feature engineering -> cleans text + unpacks JSON fields --> returns table for nlp notebook preprocessing."""
from __future__ import annotations

import ast
import json
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_TEXT_PART_COLS = ["sd_title", "item_name", "keyword", "brand"]
_SENTIMENT_LABELS = ("POSITIVE", "MIXED", "NEGATIVE")


def _to_string(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def normalize_listing_text(row: pd.Series) -> Dict[str, str]:
    parts: List[str] = []
    for col in _TEXT_PART_COLS:
        part = _to_string(row.get(col))
        if part:
            parts.append(part)
    text_raw = " ".join(parts).strip()

    text_clean = text_raw.lower()
    text_clean = re.sub(r"[^a-z0-9\s]+", " ", text_clean)
    text_clean = re.sub(r"\s+", " ", text_clean).strip()

    return {"text_raw": text_raw, "text_clean": text_clean}  # keep raw and lowercase versions


def tokenize_image_list(raw_value: object) -> List[str]:
    parsed = _coerce_json_like(raw_value)
    if not isinstance(parsed, list):
        return []
    variants: List[str] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        variant = _to_string(entry.get("variant"))
        if variant:
            variants.append(variant.lower())
    return variants


def unpack_rating_distribution(raw_value: object) -> Dict[str, Optional[float]]:
    parsed = _coerce_json_like(raw_value)
    buckets = {f"rating_pct_{i}": np.nan for i in range(1, 6)}
    if not isinstance(parsed, list):
        return buckets
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        try:
            rating = int(entry.get("rating"))
        except Exception:
            continue
        pct = _safe_float(entry.get("distribution"))
        if rating in range(1, 6) and pct is not None:
            buckets[f"rating_pct_{rating}"] = pct
    return buckets


def unpack_sentiments(raw_value: object) -> Dict[str, int]:
    parsed = _coerce_json_like(raw_value)
    counts = {f"sentiment_count_{label.lower()}": 0 for label in _SENTIMENT_LABELS}
    if not isinstance(parsed, list):
        return counts
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        label = _to_string(entry.get("sentiment")).upper()
        if label in _SENTIMENT_LABELS:
            counts[f"sentiment_count_{label.lower()}"] += 1
    return counts


def tag_missing_fields(row: pd.Series) -> Dict[str, int]:
    return {
        "flag_missing_keyword": int(not _to_string(row.get("keyword"))),
        "flag_missing_brand": int(not _to_string(row.get("brand"))),
        "flag_missing_sd_title": int(not _to_string(row.get("sd_title"))),
    }


def build_preprocessed_frame(base_df: pd.DataFrame, enriched_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        base_df,
        enriched_df,
        on=["asin", "keyword"],
        how="left",
        suffixes=("_base", "_enriched"),
    )

    for col in [
        "item_name",
        "brand",
        "bsr_best",
        "sd_title",
        "sd_customer_sentiments",
        "sd_ratings_distribution",
        "sd_number_bought_past_month",
        "image_list",
    ]:
        base_col = f"{col}_base"
        enriched_col = f"{col}_enriched"
        if base_col in merged.columns and enriched_col in merged.columns:
            merged[col] = merged[enriched_col].fillna(merged[base_col])
        elif base_col in merged.columns:
            merged[col] = merged[base_col]
        elif enriched_col in merged.columns:
            merged[col] = merged[enriched_col]
        else:
            merged[col] = np.nan

    text_parts = merged.apply(normalize_listing_text, axis=1, result_type="expand")
    tokens = merged["image_list"].apply(tokenize_image_list)
    rating_df = merged["sd_ratings_distribution"].apply(unpack_rating_distribution).apply(pd.Series)
    sentiment_df = merged["sd_customer_sentiments"].apply(unpack_sentiments).apply(pd.Series)
    flag_df = merged.apply(tag_missing_fields, axis=1, result_type="expand")

    processed = pd.concat(
        [
            merged[["asin", "keyword", "item_name", "brand", "bsr_best", "sd_number_bought_past_month"]],
            text_parts,
            rating_df,
            sentiment_df,
            flag_df,
        ],
        axis=1,
    )
    processed["image_token_count"] = tokens.apply(len)

    ordered_cols = [
        "asin",
        "keyword",
        "item_name",
        "brand",
        "bsr_best",
        "sd_number_bought_past_month",
        "text_raw",
        "text_clean",
    ]
    ordered_cols += [f"rating_pct_{i}" for i in range(1, 6)]
    ordered_cols += [
        "sentiment_count_positive",
        "sentiment_count_mixed",
        "sentiment_count_negative",
        "flag_missing_keyword",
        "flag_missing_brand",
        "flag_missing_sd_title",
        "image_token_count",
    ]

    processed = processed.reindex(columns=ordered_cols)
    processed = processed.drop_duplicates(subset=["asin", "keyword"]).reset_index(drop=True)
    return processed


def _coerce_json_like(value: object) -> object:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None  # no value to parse
    if isinstance(value, (dict, list)):
        return value  # already parsed
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(value)
            except Exception:
                return None
    return None


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(str(value).replace("%", ""))  # remove percent sign
    except Exception:
        return None  # fallback to NaN
