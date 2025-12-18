"""Combine metadata files from different assays into a single file."""

import pandas as pd
from pathlib import Path
from typing import List, Union
from loguru import logger


def combine_metadata_files(
    metadata_files: List[Union[str, Path]],
    output_path: Union[str, Path],
) -> pd.DataFrame:
    """
    Combine multiple metadata CSV files into a single file.

    Handles metadata files with different columns by filling missing columns
    with empty strings to ensure all samples have the same columns.

    Parameters:
    - metadata_files: List of paths to metadata CSV files
    - output_path: Path to save the combined metadata file

    Returns:
    - Combined DataFrame with all metadata
    """
    if not metadata_files:
        raise ValueError("No metadata files provided")

    # Read all metadata files
    dataframes = []
    for file_path in metadata_files:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"Metadata file not found: {file_path}")
            continue
        logger.info(f"Reading metadata file: {file_path}")
        df = pd.read_csv(file_path)
        dataframes.append(df)

    if not dataframes:
        raise ValueError("No valid metadata files found")

    # Get all unique columns across all dataframes
    all_columns = set()
    for df in dataframes:
        all_columns.update(df.columns)

    # Add missing columns to each dataframe with empty/NA values
    for df in dataframes:
        for col in all_columns:
            if col not in df.columns:
                df[col] = ""

    # Define preferred column order: sample_id first, then alphabetically, then r1/r2 columns last
    priority_cols = ["sample_id"]
    deprioritized_cols = sorted([col for col in all_columns if "r1" in col.lower() or "r2" in col.lower()])
    middle_cols = sorted([col for col in all_columns if col not in priority_cols and col not in deprioritized_cols])
    all_columns_ordered = priority_cols + middle_cols + deprioritized_cols

    # Ensure all dataframes have the same column order
    dataframes = [df[all_columns_ordered] for df in dataframes]

    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Save combined metadata
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    logger.info(f"Combined metadata saved to {output_path}")
    logger.info(f"Total samples: {len(combined_df)}")
    if "assay" in combined_df.columns:
        logger.info(f"Samples by assay:\n{combined_df['assay'].value_counts()}")

    return combined_df


def find_metadata_files(data_dir: Union[str, Path]) -> List[Path]:
    """
    Find all metadata CSV files in a directory.

    Looks for files matching the pattern metadata_*.csv

    Parameters:
    - data_dir: Directory to search for metadata files

    Returns:
    - List of paths to metadata files
    """
    data_dir = Path(data_dir)
    metadata_files = sorted(data_dir.glob("metadata_*.csv"))
    logger.info(f"Found {len(metadata_files)} metadata files in {data_dir}")
    return metadata_files
