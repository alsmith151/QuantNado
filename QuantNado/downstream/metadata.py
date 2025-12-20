from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import xarray as xr


def extract_metadata(ds: xr.Dataset) -> Dict[str, Any]:
    """Extract sample-level metadata with assay as the leading column when present."""

    sample_labels = ds.attrs.get("sample_names", ds.sample.values.astype(str))

    # Start with sample column; add other columns only when lengths match samples.
    metadata_df = pd.DataFrame({"sample": sample_labels})

    assay_by_sample = ds.attrs.get("assay_by_sample")
    if assay_by_sample is not None and len(assay_by_sample) == len(sample_labels):
        metadata_df["assay"] = assay_by_sample

    metadata_cols = {
        k.replace("metadata_", ""): v
        for k, v in ds.attrs.items()
        if k.startswith("metadata_") and hasattr(v, "__len__") and len(v) == len(sample_labels)
    }
    for col, values in metadata_cols.items():
        metadata_df[col] = values

    # Reorder so assay (if present) is first, followed by sample, then remaining metadata.
    front_cols = []
    if "assay" in metadata_df.columns:
        front_cols.append("assay")
    front_cols.append("sample")
    remaining = [c for c in metadata_df.columns if c not in {"assay", "sample"}]
    metadata_df = metadata_df[front_cols + remaining]

    return metadata_df
