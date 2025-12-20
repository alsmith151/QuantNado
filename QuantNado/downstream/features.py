from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd


# Lightweight GTF utilities to extract feature ranges (genes/transcripts) and
# construct promoter windows. Designed to mirror a small slice of ChIPseeker's
# annotatePeak use-case without adding heavy dependencies.


GTF_COLUMNS = [
	"seqname",
	"source",
	"feature",
	"start",
	"end",
	"score",
	"strand",
	"frame",
	"attribute",
]


def _parse_attributes(attr_str: str) -> dict:
	"""Parse the GTF attributes column into a dict of key→value strings."""

	if pd.isna(attr_str) or not attr_str:
		return {}
	parts = [p.strip() for p in attr_str.split(";") if p.strip()]
	attrs = {}
	for part in parts:
		if " " not in part:
			continue
		key, val = part.split(" ", 1)
		val = val.strip().strip('"')
		attrs[key] = val
	return attrs


def load_gtf(
	gtf_path: str | Iterable[str],
	feature_types: Optional[Iterable[str]] = None,
	usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
	"""Load a GTF file into a DataFrame with parsed attributes.

	Parameters
	----------
	gtf_path : str or iterable of str
		Path (or paths) to GTF file(s).
	feature_types : iterable of str, optional
		If provided, filter to these feature types (e.g., ["gene", "transcript"],
		["exon"], etc.).
	usecols : list[str], optional
		Additional attribute keys to extract (e.g., ["gene_id", "gene_name", "transcript_id"].
		Defaults to the common gene/transcript keys.
	"""

	if usecols is None:
		usecols = ["gene_id", "gene_name", "transcript_id", "gene_type", "gene_biotype"]

	paths = [gtf_path] if isinstance(gtf_path, str) else list(gtf_path)
	frames = []
	for path in paths:
		df = pd.read_csv(
			path,
			sep="\t",
			comment="#",
			names=GTF_COLUMNS,
			dtype={"seqname": str, "feature": str, "start": int, "end": int, "strand": str},
			usecols=["seqname", "feature", "start", "end", "strand", "attribute"],
		)
		if feature_types is not None:
			df = df[df["feature"].isin(feature_types)]

		attr_dicts = df["attribute"].apply(_parse_attributes)
		for key in usecols:
			df[key] = attr_dicts.apply(lambda d: d.get(key))

		df = df.drop(columns=["attribute"])
		frames.append(df)

	if not frames:
		return pd.DataFrame(columns=["seqname", "feature", "start", "end", "strand", *usecols])

	return pd.concat(frames, ignore_index=True)


def extract_feature_ranges(
	gtf_df: pd.DataFrame, feature_type: str = "gene"
) -> pd.DataFrame:
	"""Return ranges for a specific feature type (e.g., gene, transcript, exon)."""

	cols = [c for c in gtf_df.columns if c not in {"attribute"}]
	subset = gtf_df[gtf_df["feature"] == feature_type][cols].copy()
	subset = subset.rename(columns={"seqname": "contig"})
	return subset.reset_index(drop=True)


def extract_promoters(
	gtf_df: pd.DataFrame,
	upstream: int = 1000,
	downstream: int = 200,
	anchor_feature: str = "gene",
) -> pd.DataFrame:
	"""Build promoter windows around TSS of genes/transcripts.

	Parameters
	----------
	gtf_df : DataFrame
		DataFrame from load_gtf.
	upstream : int
		Bases upstream of TSS.
	downstream : int
		Bases downstream of TSS.
	anchor_feature : str
		Feature type to anchor on ("gene" or "transcript").
	"""

	anchors = extract_feature_ranges(gtf_df, feature_type=anchor_feature)
	if anchors.empty:
		return anchors

	starts = anchors["start"].to_numpy()
	ends = anchors["end"].to_numpy()
	strands = anchors["strand"].fillna("+").to_numpy()

	tss = starts.copy()
	tss[strands == "-"] = ends[strands == "-"]

	promo_start = tss - upstream
	promo_end = tss + downstream

	promoters = anchors.copy()
	promoters["start"] = promo_start.clip(min=0)
	promoters["end"] = promo_end
	promoters["feature"] = "promoter"
	return promoters.reset_index(drop=True)


def annotate_intervals(
	intervals: pd.DataFrame,
	feature_df: pd.DataFrame,
	feature_prefix: str = "feature_",
	require_overlap: bool = True,
) -> pd.DataFrame:
	"""Annotate intervals with overlapping feature records.

	Parameters
	----------
	intervals : DataFrame
		Must contain columns ["contig", "start", "end"].
	feature_df : DataFrame
		Feature table from extract_feature_ranges or extract_promoters; must contain
		["contig", "start", "end"].
	feature_prefix : str
		Prefix to add to feature columns in the output.
	require_overlap : bool
		If True, return only overlapping rows. If False and no overlap is found,
		returns the original intervals with NaNs for feature columns.
	"""

	required_cols = {"contig", "start", "end"}
	if not required_cols.issubset(intervals.columns):
		raise ValueError("intervals must have columns: contig, start, end")
	if not required_cols.issubset(feature_df.columns):
		raise ValueError("feature_df must have columns: contig, start, end")

	feature_cols = [c for c in feature_df.columns if c not in required_cols]
	out_rows = []

	# Simple per-contig overlap scan (sufficient for moderate sizes). For large
	# datasets consider installing pyranges and swapping this with a join.
	for contig, group in intervals.groupby("contig"):
		feats = feature_df[feature_df["contig"] == contig]
		if feats.empty:
			if not require_overlap:
				for _, row in group.iterrows():
					out_row = row.to_dict()
					for c in feature_cols:
						out_row[f"{feature_prefix}{c}"] = pd.NA
					out_rows.append(out_row)
			continue

		feat_starts = feats["start"].to_numpy()
		feat_ends = feats["end"].to_numpy()

		for _, row in group.iterrows():
			r_start, r_end = int(row["start"]), int(row["end"])
			overlaps = (feat_starts < r_end) & (feat_ends > r_start)
			if overlaps.any():
				for feat_idx in feats.index[overlaps]:
					out_row = row.to_dict()
					feat_row = feats.loc[feat_idx]
					for c in feature_cols:
						out_row[f"{feature_prefix}{c}"] = feat_row[c]
					out_rows.append(out_row)
			elif not require_overlap:
				out_row = row.to_dict()
				for c in feature_cols:
					out_row[f"{feature_prefix}{c}"] = pd.NA
				out_rows.append(out_row)

	return pd.DataFrame(out_rows)

