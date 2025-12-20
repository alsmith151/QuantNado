from __future__ import annotations

import numpy as np
import pyranges as pr
import dask.array as da


def merge_ranges(ranges: pr.PyRanges, columns: list[str] | None = None, strand: bool | None = None, sep: str = ";") -> pr.PyRanges:
    """Merge a PyRanges object and concatenate selected annotation columns per cluster."""
    if not isinstance(ranges, pr.PyRanges):
        raise ValueError("ranges must be a PyRanges object")

    cols = columns or ["Gene_name", "Gene_type", "Gene_id", "Tag", "Level"]

    clustered = ranges.cluster(strand=strand)
    merged = clustered.merge(by="Cluster", strand=strand)

    for col in cols:
        values = []
        for cluster_id in merged.Cluster.values:
            mask = clustered.Cluster.values == cluster_id
            joined = sep.join([str(v) for v in getattr(clustered, col).values[mask]])
            values.append(joined)
        setattr(merged, col, np.asarray(values))

    return merged


def get_fixed_windows(contig_lengths: dict[str, int], window_size: int = 50_000) -> pr.PyRanges:
    """Create non-overlapping fixed windows per contig length."""
    chroms = list(contig_lengths.keys())
    starts = [0] * len(chroms)
    ends = [contig_lengths[c] for c in chroms]
    gr = pr.from_dict({"Chromosome": chroms, "Start": starts, "End": ends})
    gr = gr.window(window_size, strand=False)
    gr.Ranges_ID = np.arange(len(gr))
    return gr


def masked_array_fromranges(positions_array: da.Array | np.ndarray, contig: str, ranges: pr.PyRanges) -> np.ndarray:
    """Return boolean mask for positions indicating overlap with ranges on a contig."""
    if not isinstance(positions_array, (da.Array, np.ndarray)) or not np.issubdtype(positions_array.dtype, np.integer):
        raise ValueError("positions_array must be an integer array")
    if not np.all(np.diff(np.asarray(positions_array)) > 0):
        raise ValueError("positions_array must be monotonically increasing")
    if not isinstance(ranges, pr.PyRanges):
        raise ValueError("ranges must be a PyRanges object")

    max_pos = int(np.max(positions_array))
    mask = np.zeros(max_pos + 1, dtype=bool)

    try:
        df = ranges.dfs[contig]
    except KeyError:
        return mask[positions_array]

    for start, end in zip(df.Start.values, df.End.values):
        mask[start:end] = True

    return mask[positions_array]


def default_position_mask(positions_array: da.Array | np.ndarray) -> np.ndarray:
    """Return an all-False mask for validated integer, increasing positions."""
    if not isinstance(positions_array, (da.Array, np.ndarray)) or not np.issubdtype(positions_array.dtype, np.integer):
        raise ValueError("positions_array must be an integer array")
    if not np.all(np.diff(np.asarray(positions_array)) > 0):
        raise ValueError("positions_array must be monotonically increasing")
    return np.zeros(len(positions_array), dtype=bool)


def ranges_loader(
    ranges: pr.PyRanges | list[pr.PyRanges],
    ranges_are_1based: bool = False,
    merge_intervals: bool = False,
) -> pr.PyRanges:
    """Normalize input ranges: concat, optionally shift from 1-based, reject stranded, optionally merge."""
    if isinstance(ranges, list):
        if not ranges:
            raise ValueError("No ranges provided")
        ranges = pr.concat(ranges)
        ranges.Ranges_ID = np.arange(len(ranges))

    if not isinstance(ranges, pr.PyRanges):
        raise ValueError("ranges must be a PyRanges object or list of them")

    if ranges_are_1based:
        ranges.Start = ranges.Start - 1
        ranges.End = ranges.End - 1

    if getattr(ranges, "stranded", False):
        raise ValueError("Stranded ranges not supported; drop Strand first")

    if merge_intervals:
        ranges = ranges.merge()

    return ranges