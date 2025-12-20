from __future__ import annotations

import pandas as pd
import dask.array as da
import numpy as np
import seaborn as sns
import xarray as xr
from dask_ml.decomposition import PCA

__all__ = [
    "run_pca",
    "plot_pca_scree",
    "plot_pca_scatter",
]


def _normalise_orientation(arr: xr.DataArray) -> xr.DataArray:
    """Ensure samples are the first axis; transpose if needed."""

    if arr.ndim != 2:
        raise ValueError("input_array must be 2D (samples x features)")

    sample_dim_candidates = {"sample", "sample_id"}
    dims = list(arr.dims)

    if dims[0] in sample_dim_candidates:
        return arr
    if dims[1] in sample_dim_candidates:
        return arr.transpose()

    # Unknown naming: fall back to keeping the original ordering but make intent clear.
    raise ValueError(
        "Could not identify sample dimension; expected one of ['sample', 'sample_id']"
    )


def _drop_any_nan_columns(array):
    """Drop feature columns that contain any NaNs."""

    is_dask = isinstance(array, da.Array)
    nan_mask = da.isnan(array) if is_dask else np.isnan(array)
    has_nan = nan_mask.any(axis=0)
    keep = ~has_nan
    return array[:, keep]


def _mean_impute(array):
    """Impute NaNs with column means after removing columns mostly NaN."""

    is_dask = isinstance(array, da.Array)
    nan_mask = da.isnan(array) if is_dask else np.isnan(array)
    # Remove columns where more than half the rows are NaN.
    majority_nan = nan_mask.sum(axis=0) > 0.5 * array.shape[0]
    keep = ~majority_nan
    array = array[:, keep]
    nan_mask = nan_mask[:, keep]

    mean = da.nanmean(array, axis=0) if is_dask else np.nanmean(array, axis=0)
    filled = (
        da.where(nan_mask, mean, array) if is_dask else np.where(nan_mask, mean, array)
    )
    return filled


def run_pca(
    input_array: xr.DataArray,
    n_components: int = 2,
    nan_handling_strategy: str = "drop",
    standardize: bool = False,
    random_state: int | None = None,
    subset_size: int | None = None,
    subset_strategy: str = "random",
    **kwargs,
):
    """Run PCA on a 2D xarray.DataArray with samples x features.

    Parameters
    ----------
    input_array : xr.DataArray
        2D array with sample dimension and feature/context dimension.
    n_components : int
        Number of principal components to return.
    nan_handling_strategy : str
        One of ['drop', 'set_to_zero', 'mean_value_imputation'].
    standardize : bool
        If True, z-score features before PCA.
    random_state : int | None
        Optional random seed for PCA (where supported by the solver/backend).
    subset_features : slice | np.ndarray | None
        Optional indexer along the feature axis to select a subset before PCA.
    subset_size : int | None
        If provided and smaller than the feature count, select this many features
        (randomly or from the start, depending on subset_strategy) before PCA.
    subset_strategy : str
        "random" (default) samples features without replacement using random_state,
        or "first" keeps the first subset_size features.
    **kwargs
        Additional arguments passed to PCA constructor.
    """

    arr_2d = _normalise_orientation(input_array)
    array = arr_2d.data
    if subset_size is not None:
        n_features = array.shape[1]
        if subset_size < n_features:
            if subset_strategy not in {"random", "first"}:
                raise ValueError("subset_strategy must be 'random' or 'first'")
            if subset_strategy == "first":
                subset_idx = np.arange(subset_size)
            else:
                rng = np.random.default_rng(random_state)
                subset_idx = rng.choice(n_features, subset_size, replace=False)
            array = array[:, subset_idx]

    match nan_handling_strategy:
        case "drop":
            array = _drop_any_nan_columns(array)
        case "set_to_zero":
            array = (
                da.nan_to_num(array)
                if isinstance(array, da.Array)
                else np.nan_to_num(array)
            )
        case "mean_value_imputation":
            array = _mean_impute(array)
        case _:
            raise ValueError(
                "nan_handling_strategy must be one of ['drop', 'set_to_zero', 'mean_value_imputation']"
            )

    # Optional standardization
    if standardize:
        if isinstance(array, da.Array):
            mean = da.nanmean(array, axis=0)
            std = da.nanstd(array, axis=0)
            array = (array - mean) / da.maximum(std, 1e-12)
        else:
            if not isinstance(array, np.ndarray):
                array = np.asarray(array)
            mean = np.nanmean(array, axis=0)
            std = np.nanstd(array, axis=0)
            array = (array - mean) / np.maximum(std, 1e-12)

    pca_kwargs = {"n_components": n_components, **kwargs}
    if random_state is not None:
        pca_kwargs["random_state"] = random_state

    pca = PCA(**pca_kwargs)
    array.compute_chunk_sizes()

    pca_object = pca.fit(array)
    transformed = pca_object.transform(array)

    return pca_object, transformed


def plot_pca_scree(pca_object, filepath: str | None = None, **kwargs):
    """Bar plot of explained variance ratios."""

    exp_var = np.asarray(pca_object.explained_variance_ratio_)
    labels = [f"PC{i + 1} ({exp_var[i]:.2f})" for i in range(exp_var.size)]
    plot = sns.catplot(x=labels, y=exp_var, kind="bar", **kwargs)
    plot.set_ylabels("Explained variance")
    plot.set_xlabels("Principal components")
    if filepath is not None:
        plot.savefig(filepath)
    return plot


def plot_pca_scatter(
    pca_object,
    pca_result,
    xaxis_pc: int = 1,
    yaxis_pc: int = 2,
    metadata_df: xr.DataArray | None = None,
    colour_by: str | None = None,
    shape_by: str | None = None,
    filepath: str | None = None,
    sample_column: str = "sample_id",
    **kwargs,
):
    """Scatter plot of two principal components."""

    pca_result_arr = np.asarray(pca_result)

    exp_var = np.asarray(pca_object.explained_variance_ratio_)
    xkey, ykey = xaxis_pc - 1, yaxis_pc - 1
    xlabel = f"PC{xaxis_pc} ({exp_var[xkey]:.2%})"
    ylabel = f"PC{yaxis_pc} ({exp_var[ykey]:.2%})"
    hue = (
        metadata_df[colour_by]
        if colour_by and metadata_df is not None and colour_by in metadata_df.columns
        else None
    )
    style_vals = (
        metadata_df[shape_by]
        if shape_by and metadata_df is not None and shape_by in metadata_df.columns
        else None
    )
    palette = None
    if hue is not None:
        unique_vals = pd.unique(hue)
        palette = dict(zip(unique_vals, sns.color_palette("tab10", len(unique_vals))))
    sample_labels = None
    if metadata_df is not None:
        if sample_column in metadata_df.columns:
            sample_labels = metadata_df[sample_column]
        elif "sample" in metadata_df.columns:
            sample_labels = metadata_df["sample"]
        elif hasattr(metadata_df, "index") and len(metadata_df.index) == pca_result_arr.shape[0]:
            sample_labels = metadata_df.index
    if sample_labels is not None and len(sample_labels) != pca_result_arr.shape[0]:
        sample_labels = None
    plot = sns.relplot(
        x=pca_result_arr[:, xkey],
        y=pca_result_arr[:, ykey],
        kind="scatter",
        palette=palette if palette is not None else None,
        hue=hue if hue is not None else None,
        style=style_vals if style_vals is not None else None,
        **kwargs,
    )
    if sample_labels is not None:
        for i, label in enumerate(sample_labels):
            plot.ax.annotate(
                str(label),
                (pca_result_arr[i, xkey], pca_result_arr[i, ykey]),
                fontsize=8,
                alpha=0.7,
                ha="left",
                va="bottom",
                xytext=(4, 4),
                textcoords="offset points",
            )
    
    plot.set_xlabels(xlabel)
    plot.set_ylabels(ylabel)
    if filepath is not None:
        plot.savefig(filepath)
    return plot
