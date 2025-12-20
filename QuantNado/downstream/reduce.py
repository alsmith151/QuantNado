from __future__ import annotations

import numpy as np
import dask.array as da
import xarray as xr


def _ensure_dask_2d(data: xr.DataArray | np.ndarray | da.Array) -> da.Array:
    """Return a 2D dask array (positions x samples) for reduction."""
    arr = data
    if isinstance(arr, xr.DataArray):
        arr = arr.data
    if not isinstance(arr, da.Array):
        arr = da.from_array(arr)
    if arr.ndim != 2:
        raise ValueError("expected a 2D array (positions x samples)")
    # Let dask pick sensible position chunks; keep sample chunks as-is.
    arr = arr.rechunk({0: "auto"})
    return arr


def _reduce_byranges_prefix(
    row_starts: np.ndarray,
    row_ends: np.ndarray,
    data: xr.DataArray | np.ndarray | da.Array,
    *,
    min_count: int = 1,
) -> dict[str, da.Array]:
    """
    Reduce ranges via prefix sums (no numba, no external deps).

    Assumes row indices are 0-based, end-exclusive, and len(row_starts)==len(row_ends).
    Works for both float (NaN-aware) and integer data.
    """

    if row_starts.shape != row_ends.shape:
        raise ValueError("row_starts and row_ends must have the same shape")

    starts = np.asarray(row_starts, dtype=np.int64)
    ends = np.asarray(row_ends, dtype=np.int64)

    arr = _ensure_dask_2d(data)

    is_float = np.issubdtype(arr.dtype, np.floating)
    # Prepare value and mask arrays for prefix sums.
    values = da.nan_to_num(arr, nan=0.0) if is_float else arr
    mask = (~da.isnan(arr)) if is_float else da.ones_like(arr, dtype=np.int64)

    # Prefix sums along positions (axis 0). Prepend a zero row so end can equal len.
    sum_pref = da.concatenate(
        [da.zeros((1, arr.shape[1]), dtype=values.dtype), da.cumsum(values, axis=0)],
        axis=0,
    )
    count_pref = da.concatenate(
        [da.zeros((1, arr.shape[1]), dtype=np.int64), da.cumsum(mask, axis=0)],
        axis=0,
    )

    # Gather prefix rows for starts/ends; da.take handles dask-aware indexing.
    sum_start = da.take(sum_pref, starts, axis=0)
    sum_end = da.take(sum_pref, ends, axis=0)
    count_start = da.take(count_pref, starts, axis=0)
    count_end = da.take(count_pref, ends, axis=0)

    sums = sum_end - sum_start
    counts = count_end - count_start

    # Mean with minimum count threshold.
    means = da.ma.filled(
        sums / da.ma.masked_less(counts, min_count),
        np.nan,
    )

    return {"sum": sums, "count": counts.astype(np.int64), "mean": means}


def reduce_byranges_signal(
    signal: xr.DataArray,
    ranges_df=None,
    bed_file: str | None = None,
    start_col: str = "start",
    end_col: str = "end",
    contig_col: str | None = None,
    min_count: int = 1,
    reduction: str = "mean",
) -> xr.Dataset:
    """
    Summarize a flattened signal over genomic ranges using dask prefix sums.

    Parameters
    ----------
    signal : xr.DataArray
        Must have dims including "position_flat" and "sample".
    ranges_df : pandas.DataFrame, optional
        Contains 0-based, end-exclusive ranges. Required if bed_file is not provided.
    bed_file : str | None
        Optional path to BED file to load ranges from instead of ranges_df.
    start_col, end_col : str
        Column names for start/end.
    contig_col : str | None
        Optional contig column to carry through as a coordinate.
    min_count : int
        Minimum count required to report a mean; otherwise NaN.
    """

    if "position_flat" not in signal.dims or "sample" not in signal.dims:
        raise ValueError("signal must have dims including 'position_flat' and 'sample'")

    arr = signal.transpose("position_flat", "sample")

    if bed_file is not None:
        import pandas as pd

        bed_cols = ["contig", "start", "end"]
        # import just the first 3 columns
        ranges_df = pd.read_csv(bed_file, sep="\t", header=None, names=bed_cols, usecols=[0,1,2])
        start_col = "start"
        end_col = "end"
        contig_col = "contig"
    elif ranges_df is None:
        raise TypeError("ranges_df is required when bed_file is not provided")

    starts = np.asarray(ranges_df[start_col], dtype=np.int64)
    ends = np.asarray(ranges_df[end_col], dtype=np.int64)

    arr_len = int(arr.shape[0])
    # Clip to array bounds and drop invalid ranges
    starts = starts.clip(min=0)
    ends = ends.clip(max=arr_len)
    valid = (ends > starts) & (starts < arr_len) & (ends > 0)
    if not np.all(valid):
        starts = starts[valid]
        ends = ends[valid]
        ranges_df = ranges_df.loc[valid]
    if starts.size == 0:
        raise ValueError("No valid ranges after clipping to signal length")

    reduced = _reduce_byranges_prefix(starts, ends, arr, min_count=min_count)

    arr_dask = _ensure_dask_2d(arr)

    def _stack_reduction(op):
        results = [op(arr_dask[start:end], axis=0) for start, end in zip(starts, ends)]
        return da.stack(results, axis=0)

    if reduction == "mean":
        reduction_data = reduced["mean"]
    elif reduction == "max":
        reduction_data = _stack_reduction(da.max)
    elif reduction == "min":
        reduction_data = _stack_reduction(da.min)
    elif reduction == "median":
        reduction_data = _stack_reduction(lambda x, axis: da.percentile(x, 50, axis=axis))
    else:
        raise ValueError("reduction must be one of ['mean', 'max', 'min', 'median']")

    coords: dict[str, object] = {
        "ranges": np.arange(starts.size, dtype=int),
        "sample": arr.sample.values,
        "start": ("ranges", starts),
        "end": ("ranges", ends),
        "range_length": ("ranges", ends - starts),
    }
    if contig_col and contig_col in ranges_df.columns:
        coords["contig"] = (
            "ranges",
            np.asarray(ranges_df[contig_col]),
        )

    return xr.Dataset(
        {
            "sum": (("ranges", "sample"), reduced["sum"]),
            "count": (("ranges", "sample"), reduced["count"]),
            "mean": (("ranges", "sample"), reduced["mean"]),
            reduction: (("ranges", "sample"), reduction_data),
        },
        coords=coords,
    )