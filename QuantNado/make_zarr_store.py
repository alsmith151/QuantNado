import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import bamnado
import numpy as np
import pandas as pd
import sparse
import xarray as xr
import zarr
from loguru import logger

# Constants for bamnado parameters
BIN_SIZE = 1
SCALE_FACTOR = 1.0
USE_FRAGMENT = False
IGNORE_SCAFFOLD_CHROMS = True


class BamZarrStore:
    """
    Context manager for creating and managing BAM-to-Zarr conversions.

    Handles:
    - Store creation and cleanup
    - Thread-safe chromosome appending
    - Metadata management
    - Error handling and cleanup on failure
    """

    def __init__(self, output_path: Path, sample_name: str, overwrite: bool = True):
        self.output_path = Path(output_path)
        self.sample_name = sample_name
        self.overwrite = overwrite
        self.root = None
        self.write_lock = Lock()
        self.sparsity_values = []

    def __enter__(self):
        """Initialize the zarr store."""
        # Clean up existing store if overwrite is True
        if self.overwrite and self.output_path.exists():
            shutil.rmtree(self.output_path)
            logger.info(f"Removed existing zarr file: {self.output_path}")

        # Create zarr root group (Zarr v3 API - no need for DirectoryStore)
        self.root = zarr.open_group(self.output_path, mode="w")

        logger.info(f"Created zarr store: {self.output_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up and finalize the zarr store."""
        if exc_type:
            # Error occurred - clean up
            logger.error(f"Error during processing: ({exc_type}): {exc_val}")
            logger.warning(f"Cleaning up incomplete store: {self.output_path}")
            if self.output_path.exists():
                shutil.rmtree(self.output_path)
            return False  # Re-raise the exception

        # Success - finalize metadata
        if self.sparsity_values:
            avg_sparsity = np.mean(self.sparsity_values)
            self.root.attrs["sample"] = self.sample_name
            self.root.attrs["description"] = "BAM coverage data across all chromosomes"
            self.root.attrs["average_sparsity"] = f"{avg_sparsity:.2f}%"
            self.root.attrs["num_chromosomes"] = len(self.sparsity_values)

            logger.info(f"Dataset created (avg sparsity: {avg_sparsity:.2f}%)")

        logger.info(f"Zarr store finalized: {self.output_path}")
        return True

    def append_chromosome(self, contig: str, ds_chrom: xr.Dataset, sparsity: float):
        """
        Thread-safe append of a chromosome dataset.

        Parameters:
        - contig: Chromosome name
        - ds_chrom: xarray Dataset for this chromosome
        - sparsity: Sparsity percentage for logging
        """
        with self.write_lock:
            encoding = {"signal": {"chunks": (100_000,)}}
            ds_chrom.to_zarr(
                self.output_path,
                mode="a",
                group=contig,
                encoding=encoding,
                consolidated=False,
            )
            self.sparsity_values.append(sparsity)
            logger.debug(f"Appended {contig} to store")


def _select_optimal_dtype(max_val: float, contig: str) -> tuple[np.dtype, str]:
    """
    Select optimal dtype based on maximum value to minimize storage.

    Parameters:
    - max_val: Maximum value in the signal array
    - contig: Chromosome name (for logging)

    Returns:
    - dtype: NumPy dtype to use
    - dtype_name: String name of the dtype
    """
    if max_val <= 65535:
        # Use uint16 for 50% space savings
        return np.uint16, "uint16"
    elif max_val <= 4294967295:
        # Use uint32 if values exceed uint16 range
        logger.info(
            f"  {contig}: Using uint32 (max coverage {max_val:.0f} exceeds uint16 range)"
        )
        return np.uint32, "uint32"
    else:
        # Use float32 if values exceed uint32 range (extremely rare)
        logger.warning(
            f"  {contig}: Using float32 (max coverage {max_val:.0f} exceeds uint32 range)"
        )
        return np.float32, "float32"


def process_chromosome(
    bam_file: str,
    contig: str,
    chrom_size: int,
) -> tuple[str, xr.Dataset, float]:
    """
    Worker function to process a single chromosome.

    Returns:
    - contig name
    - Dataset for this chromosome
    - Sparsity percentage
    """
    # Extract signal
    signal = bamnado.get_signal_for_chromosome(
        bam_path=bam_file,
        chromosome_name=contig,
        bin_size=BIN_SIZE,
        scale_factor=SCALE_FACTOR,
        use_fragment=USE_FRAGMENT,
        ignore_scaffold_chromosomes=IGNORE_SCAFFOLD_CHROMS,
    )

    # Calculate sparsity
    signal_sparse = sparse.COO.from_numpy(signal)
    sparsity = 100 * (1 - signal_sparse.nnz / signal.size)

    # Dynamically choose dtype based on max value
    max_val = signal.max()
    dtype, dtype_name = _select_optimal_dtype(max_val, contig)

    logger.info(
        f"  {contig}: {sparsity:.2f}% sparse (max: {max_val:.0f}, dtype: {dtype_name})"
    )

    # Convert signal to chosen dtype
    signal_encoded = signal.astype(dtype)

    # Create dataset for this chromosome
    ds_chrom = xr.Dataset(
        {"signal": (["position"], signal_encoded)},
        coords={
            "position": np.arange(chrom_size),
            "chromosome": contig,
        },
    )

    return contig, ds_chrom, sparsity


def process_bam(
    bam_file: str,
    chromsizes: str | Path | dict,
    cache_dir: Path,
    max_workers: int = 4,
) -> Path:
    """
    Process a single BAM file for all chromosomes in parallel using BamZarrStore.

    Uses parallel processing with thread-safe incremental appending to avoid
    slow concatenation - each chromosome is appended to the zarr store as soon
    as it's processed.

    Parameters:
    - bam_file: Path to the BAM file.
    - chromsizes: Either a path to a chrom.sizes file or a dictionary mapping chromosome names to sizes.
    - cache_dir: Directory to store cached Zarr files.
    - max_workers: Number of parallel threads to use (default: 4).

    Returns:
    - Path to the cached Zarr file.
    """
    # Parse chromsizes if it's a file path
    if isinstance(chromsizes, (str, Path)):
        chromsizes_dict = {}
        with open(chromsizes) as f:
            for line in f:
                chrom, size = line.strip().split()
                # Only include main chromosomes (chr1-22, X, Y, M)
                if chrom.startswith("chr") and "_" not in chrom:
                    chromsizes_dict[chrom] = int(size)
        logger.info(f"Loaded {len(chromsizes_dict)} chromosomes from {chromsizes}")
        chromsizes = chromsizes_dict

    sample_name = Path(bam_file).stem
    cache_file = cache_dir / f"{sample_name}.zarr"

    logger.info(
        f"Processing BAM file: {bam_file} for {len(chromsizes)} chromosomes in parallel (max_workers={max_workers})"
    )

    # Use context manager for automatic store management
    with BamZarrStore(cache_file, sample_name, overwrite=True) as store:
        # Process chromosomes in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chromosome processing tasks
            future_to_chrom = {
                executor.submit(
                    process_chromosome, bam_file, contig, chrom_size
                ): contig
                for contig, chrom_size in chromsizes.items()
            }

            # Process results as they complete and append immediately
            for future in as_completed(future_to_chrom):
                contig, ds_chrom, sparsity = future.result()
                # Thread-safe appending handled by store.append_chromosome
                store.append_chromosome(contig, ds_chrom, sparsity)

        logger.info(f"Structure: {len(chromsizes)} chromosome groups")

    return cache_file


def combine_cached_zarrs(
    cache_dir: Path,
    metadata_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Combine cached Zarr files into a single xarray Dataset.

    Each cached zarr file has chromosomes as separate groups. This function:
    1. Loads each sample's chromosome groups
    2. Combines chromosomes within each sample
    3. Combines all samples along the "sample" dimension
    4. Adds metadata coordinates

    Parameters:
    - cache_dir: Directory containing cached Zarr files (with chromosome groups).
    - metadata_df: DataFrame containing metadata for the BAM files.
    - output_path: Path to save the combined Zarr dataset.
    """
    logger.info(f"Combining cached Zarr files from {cache_dir}")

    try:
        # Load all cached datasets
        cached_files = sorted(cache_dir.glob("*.zarr"))

        if not cached_files:
            raise ValueError(f"No zarr files found in {cache_dir}")

        logger.info(f"Found {len(cached_files)} cached Zarr files")

        sample_datasets = []
        for zarr_file in cached_files:
            try:
                logger.info(f"Loading {zarr_file.name}...")

                # Open root to get chromosome groups
                root = zarr.open_group(zarr_file, mode="r")
                chromosomes = sorted(root.group_keys())

                logger.info(f"  Found {len(chromosomes)} chromosome groups")

                # Load each chromosome group
                chrom_datasets = []
                for chrom in chromosomes:
                    ds_chrom = xr.open_zarr(zarr_file, group=chrom)
                    chrom_datasets.append(ds_chrom)

                # Combine chromosomes for this sample
                logger.info(f"  Combining chromosomes for {zarr_file.stem}...")
                sample_ds = xr.concat(chrom_datasets, dim="chromosome", join="outer")
                sample_ds = sample_ds.assign_coords(chromosome=chromosomes)

                # Copy metadata from root attrs
                for key, value in root.attrs.items():
                    if key not in ["chromosomes"]:  # Skip chromosome list
                        sample_ds.attrs[key] = value

                sample_datasets.append(sample_ds)

            except Exception as e:
                logger.warning(f"Failed to load {zarr_file}: {e}")

        if not sample_datasets:
            raise ValueError("No valid zarr datasets could be loaded")

        logger.info(f"Loaded {len(sample_datasets)} sample datasets")

        # Combine datasets along the "sample" dimension
        logger.info("Combining all samples along 'sample' dimension...")
        combined_ds = xr.concat(sample_datasets, dim="sample", join="outer")
        logger.info("Combined datasets along 'sample' dimension")

        # Add sample names as coordinates
        sample_names = [f.stem for f in cached_files]
        combined_ds = combined_ds.assign_coords(sample=sample_names)

        # Add metadata
        logger.info(f"Adding metadata from {len(metadata_df)} rows")
        for col in metadata_df.columns:
            if col != "sample_id":
                combined_ds.coords[col] = ("sample", metadata_df[col].values)

        # Save combined dataset
        logger.info(f"Saving combined dataset to {output_path}")
        encoding = {
            "signal": {"chunks": (1, 1, 100_000)}
        }  # (sample, chromosome, position)
        combined_ds.to_zarr(
            output_path, mode="w", encoding=encoding, consolidated=False
        )
        logger.info(f"Combined dataset saved successfully to {output_path}")

    except Exception as e:
        logger.error(f"Error combining zarr files: {e}")
        raise
