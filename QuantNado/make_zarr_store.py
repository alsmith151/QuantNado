import argparse
import logging

from pathlib import Path
from typing import Optional

import bamnado
import numpy as np
import pandas as pd
import sparse
import xarray as xr



# Configure logging to include a file handler
log_file = "quantnado_processing.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler(log_file),  # Logs to file
    ],
)
logger = logging.getLogger(__name__)
logger.info("Logging initialized. Logs will be written to quantnado_processing.log")


def process_bam(
    assay_type: str,
    fasta_path: str,
    contig: str,
    start: int = 0,
    end: Optional[int] = None,
    stepsize: int = 1_000_000,
    chunk_size: int = 100_000,
):
    """
    Generalized function to process assay data (e.g., methylation, ATAC, RNA).
    """
    signal = bamnado.get_signal_for_chromosome(
        bam_path=fasta_path,
        chromosome_name=contig,
        bin_size=1,
        scale_factor=1.0,
        use_fragment=False,
        ignore_scaffold_chromosomes=True,
    )
    return signal


def process_bam_files(
    bam_files: list,
    metadata_df: pd.DataFrame,
    contig: str,
    chrom_size: int,
):
    """
    Process multiple BAM files and create a single xarray Dataset.

    Parameters:
    - bam_files: List of BAM file paths.
    - metadata_df: DataFrame containing metadata for the BAM files.
    - contig: Chromosome/contig name.
    - chrom_size: Size of the chromosome/contig.

    Returns:
    - xarray Dataset with sparse arrays for BAM coverage.
    """
    signals = {}

    for bam_file in bam_files:
        sample_name = Path(bam_file).stem
        signal = bamnado.get_signal_for_chromosome(
            bam_path=bam_file,
            chromosome_name=contig,
            bin_size=1,
            scale_factor=1.0,
            use_fragment=False,
            ignore_scaffold_chromosomes=True,
        )
        signals[sample_name] = signal

    # Stack signals into a dense array
    signal_array = np.stack([signals[sample] for sample in signals], axis=0)

    # Convert to sparse array
    signal_sparse = sparse.COO.from_numpy(signal_array)

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "signal": ("sample", "position", signal_sparse),
        },
        coords={
            "sample": list(signals.keys()),
            "position": np.arange(chrom_size),
            "chromosome": contig,
        },
        attrs={
            "description": "Sparse BAM coverage data",
        },
    )

    # Add metadata
    for col in metadata_df.columns:
        if col != "sample_id":
            ds.coords[col] = ("sample", metadata_df[col].values)

    return ds


def process_and_cache_bam(
    bam_file: str,
    contig: str,
    chrom_size: int,
    cache_dir: Path,
):
    """
    Process a single BAM file and cache the result as a Zarr file.

    Parameters:
    - bam_file: Path to the BAM file.
    - contig: Chromosome/contig name.
    - chrom_size: Size of the chromosome/contig.
    - cache_dir: Directory to store cached Zarr files.

    Returns:
    - Path to the cached Zarr file.
    """
    logger.info(f"Processing BAM file: {bam_file} for contig: {contig}")
    try:
        signal = bamnado.get_signal_for_chromosome(
            bam_path=bam_file,
            chromosome_name=contig,
            bin_size=1,
            scale_factor=1.0,
            use_fragment=False,
            ignore_scaffold_chromosomes=True,
        )
        logger.info(f"Signal extracted for contig: {contig}")

        # Create sparse array
        signal_sparse = sparse.COO.from_numpy(signal)
        logger.info(f"Sparse array created for contig: {contig}")

        # Convert sparse array to dense before saving
        signal_dense = signal_sparse.todense()
        logger.info(f"Converted sparse array to dense for contig: {contig}")

        # Create xarray Dataset with dense array
        ds = xr.Dataset(
            {
                "signal": ("position", signal_dense),
            },
            coords={
                "position": np.arange(chrom_size),
                "chromosome": contig,
            },
            attrs={
                "sample": Path(bam_file).stem,
                "description": "Dense BAM coverage data",
            },
        )
        logger.info(f"xarray Dataset created with dense array for contig: {contig}")

        # Ensure sparse encoding is used when saving to Zarr
        encoding = {"signal": {"filters": None, "chunks": (1000,)}}
        logger.info(f"Using sparse encoding for contig: {contig}")

        # Save to cache with sparse encoding
        cache_file = cache_dir / f"{Path(bam_file).stem}_{contig}.zarr"
        ds.to_zarr(cache_file, mode="w", encoding=encoding)
        logger.info(f"Sparse data saved to cache: {cache_file}")

        return cache_file
    except Exception as e:
        logger.error(f"Error processing BAM file: {bam_file} for contig: {contig} - {e}")
        raise


def combine_cached_zarrs(
    cache_dir: Path,
    metadata_df: pd.DataFrame,
    output_path: Path,
):
    """
    Combine cached Zarr files into a single xarray Dataset.

    Parameters:
    - cache_dir: Directory containing cached Zarr files.
    - metadata_df: DataFrame containing metadata for the BAM files.
    - output_path: Path to save the combined Zarr dataset.
    """
    # Load all cached datasets
    cached_files = sorted(cache_dir.glob("*.zarr"))
    datasets = [xr.open_zarr(f) for f in cached_files]

    # Combine datasets along the "sample" dimension
    combined_ds = xr.concat(datasets, dim="sample")

    # Add metadata
    for col in metadata_df.columns:
        if col != "sample_id":
            combined_ds.coords[col] = ("sample", metadata_df[col].values)

    # Save combined dataset
    combined_ds.to_zarr(output_path, mode="w")
    print(f"Combined dataset saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process BAM files and create xarray Dataset."
    )
    parser.add_argument(
        "--bam-file", help="Path to a single BAM file to process and cache."
    )
    parser.add_argument(
        "--cache-dir", required=True, help="Directory to store cached Zarr files."
    )
    parser.add_argument("--contig", required=True, help="Chromosome/contig name.")
    parser.add_argument(
        "--chrom-size", type=int, required=True, help="Size of the chromosome/contig."
    )
    parser.add_argument(
        "--process-bam",
        action="store_true",
        help="Process and cache a single BAM file.",
    )
    parser.add_argument(
        "--combine-datasets",
        action="store_true",
        help="Combine cached Zarr files into a final dataset.",
    )
    parser.add_argument("--metadata-path", help="Path to metadata CSV file.")
    parser.add_argument("--output-path", help="Path to save the combined Zarr dataset.")

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.process_bam and args.bam_file:
        # Process and cache a single BAM file
        process_and_cache_bam(args.bam_file, args.contig, args.chrom_size, cache_dir)

    if args.combine_datasets:
        # Combine cached Zarr files into a final dataset
        if not args.metadata_path or not args.output_path:
            raise ValueError(
                "--metadata-path and --output-path are required for combining datasets."
            )

        metadata_df = pd.read_csv(args.metadata_path)
        combine_cached_zarrs(cache_dir, metadata_df, Path(args.output_path))


if __name__ == "__main__":
    main()
