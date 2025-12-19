import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import bamnado
import numpy as np
import pandas as pd
import sparse
import xarray as xr
import zarr
from loguru import logger
from zarr.storage import ZipStore

# Constants for bamnado parameters
BIN_SIZE = 1
SCALE_FACTOR = 1.0
USE_FRAGMENT = False
IGNORE_SCAFFOLD_CHROMS = True


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


def _parse_chromsizes(
    chromsizes: str | Path | dict, filter_chromosomes: bool = True
) -> dict:
    """
    Parse chromosome sizes from file or dict.

    Parameters:
    - chromsizes: Either a path to a chrom.sizes file or a dictionary mapping chromosome names to sizes.
    - filter_chromosomes: If True, only include main chromosomes (chr1-22, X, Y, M).

    Returns:
    - Dictionary mapping chromosome names to sizes.
    """
    if isinstance(chromsizes, dict):
        return chromsizes

    chromsizes_dict = {}
    with open(chromsizes) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                logger.warning(
                    f"Skipping invalid line {line_num} in chromsizes file: {line}"
                )
                continue

            chrom, size_str = parts[0], parts[1]

            try:
                size = int(size_str)
            except ValueError:
                logger.warning(
                    f"Skipping line {line_num} - invalid size '{size_str}': {line}"
                )
                continue

            # Filter chromosomes if requested
            if filter_chromosomes:
                if chrom.startswith("chr") and "_" not in chrom:
                    chromsizes_dict[chrom] = size
            else:
                chromsizes_dict[chrom] = size

    logger.info(f"Loaded {len(chromsizes_dict)} chromosomes from {chromsizes}")
    return chromsizes_dict


def _append_sample_to_store(
    sample_data: dict,
    sample_name: str,
    chromosomes: list,
    chromsizes_dict: dict,
    store_path: Path,
    is_first_sample: bool,
) -> None:
    """
    Append a single sample to the Zarr store.

    This version avoids expensive xarray.concat by preallocating a 2D array
    with shape (chromosome, position) and inserting chromosome signals directly.
    """

    # Determine max chromosome length for padding
    max_chrom_size = max(chromsizes_dict[chrom] for chrom in chromosomes)

    # Determine dtype from one chromosome (assuming consistent dtype)
    # You might want to generalize if sample_data chroms differ in dtype
    first_chrom = chromosomes[0]
    dtype = sample_data[first_chrom].dtype

    # Preallocate array (chromosome x position), fill with 0 or np.nan as appropriate
    # Use zeros assuming coverage signals; if missing data should be nan, use np.nan
    data = np.zeros((len(chromosomes), max_chrom_size), dtype=dtype)

    # Fill in each chromosome's data (padding shorter chromosomes)
    for i, chrom in enumerate(chromosomes):
        chrom_data = sample_data[chrom]
        chrom_len = chromsizes_dict[chrom]
        data[i, :chrom_len] = chrom_data

        # If padding needed and you want to use NaN instead of 0:
        # if chrom_len < max_chrom_size:
        #     data[i, chrom_len:] = np.nan  # Requires float dtype

    # Create Dataset with dimensions (sample, chromosome, position)
    # Since this is one sample, add sample dimension size=1
    ds_sample = xr.Dataset(
        {"signal": (["sample", "chromosome", "position"], data[np.newaxis, :, :])},
        coords={
            "sample": np.array([sample_name], dtype="object"),
            "chromosome": chromosomes,
            "position": np.arange(max_chrom_size),
        },
    )

    encoding = {
        "signal": {
            "chunks": (1, 1, 1_000_000),  # Chunk by sample, chromosome, position
        }
    }

    if is_first_sample:
        # Create new store
        ds_sample.to_zarr(
            store_path,
            mode="w",
            encoding=encoding,
            consolidated=False,
        )
    else:
        # Append to existing store
        ds_sample.to_zarr(
            store_path,
            mode="a",
            append_dim="sample",
            consolidated=False,
        )


def _finalize_store(
    store_path: Path,
    sample_id: list,
    metadata: pd.DataFrame | None,
    sparsity_values: list,
    num_chromosomes: int,
) -> None:
    """Add metadata and attributes to the final store."""

    ds = xr.open_zarr(store_path, consolidated=False)

    if metadata is not None:
        logger.info("Adding metadata coordinates...")
        sample_col = None
        for col in metadata.columns:
            if set(metadata[col].values) >= set(sample_id):
                sample_col = col
                break

        if sample_col:
            metadata = metadata.set_index(sample_col).loc[sample_id].reset_index()

            # Ensure standard dtypes for all metadata columns
            for col in metadata.columns:
                if col != sample_col:
                    values = metadata[col].values
                    # Convert object dtype to string
                    if values.dtype == object:
                        values = values.astype(str)
                    # Convert pandas nullable types to standard numpy types
                    elif pd.api.types.is_integer_dtype(values):
                        values = values.astype(int)
                    elif pd.api.types.is_float_dtype(values):
                        values = values.astype(float)
                    ds = ds.assign_coords({col: ("sample", values)})
        else:
            logger.warning("Could not find sample names in metadata DataFrame")

    avg_sparsity = np.mean(sparsity_values) if sparsity_values else 0
    ds.attrs["description"] = "BAM coverage data across all samples and chromosomes"
    ds.attrs["average_sparsity"] = f"{avg_sparsity:.2f}%"
    ds.attrs["num_samples"] = len(sample_id)
    ds.attrs["num_chromosomes"] = num_chromosomes

    ds.to_zarr(store_path, mode="a", consolidated=False)


def _compress_to_zipstore(source_dir: Path, target_zip: Path) -> None:
    """Compress a DirectoryStore to a ZipStore."""

    # Load from directory
    ds = xr.open_zarr(source_dir, consolidated=False)

    # Save to ZipStore
    with ZipStore(target_zip, mode="w") as store:
        encoding = {
            "signal": {
                "chunks": (1, 1, 1_000_000),
            }
        }
        ds.to_zarr(store, mode="w", encoding=encoding, consolidated=False)


def _process_assay(
    bam_files: list,
    sample_id: list,
    metadata: pd.DataFrame,
    chromsizes_dict: dict,
    chromosomes: list,
    temp_store_path: Path,
    final_store_path: Path,
    max_workers: int,
    overwrite: bool,
    use_zip: bool,
) -> None:
    """
    Process BAM files grouped by assay, creating (sample × chromosome × position) structure.

    Uses a flattened/long format where assay is a coordinate along the sample dimension,
    avoiding the need for padding to handle different sample counts per assay.
    """

    # Find sample column in metadata - prefer 'sample_id' column
    sample_col = None
    if "sample_id" in metadata.columns:
        sample_col = "sample_id"
    else:
        # Try to find a column that contains sample names
        for col in metadata.columns:
            if set(metadata[col].values) >= set(sample_id):
                sample_col = col
                break

    if not sample_col:
        raise ValueError(
            f"Could not find sample column in metadata DataFrame. "
            f"Looking for samples: {sample_id[:5]}... "
            f"Available metadata columns: {list(metadata.columns)}"
        )

    # Group samples by assay and track metadata name mapping
    assay_groups = {}
    sample_to_metadata = {}  # Maps BAM sample name to metadata sample name

    for bam_file, sample_name in zip(bam_files, sample_id):
        # Try exact match first
        sample_metadata = metadata[metadata[sample_col] == sample_name]

        # If not found, try fuzzy matching by removing suffix after last underscore
        # This handles ChIP cases like 'SEM-DMSO-H3K27Ac_H3K27Ac' -> 'SEM-DMSO-H3K27Ac'
        # and 'SEM-DMSO-H3K27Ac_Input' -> 'SEM-DMSO-H3K27Ac'
        metadata_sample_name = sample_name
        if sample_metadata.empty and "_" in sample_name:
            base_name = sample_name.rsplit("_", 1)[0]
            suffix = sample_name.rsplit("_", 1)[1]
            sample_metadata = metadata[metadata[sample_col] == base_name]

            if not sample_metadata.empty:
                # For ChIP samples, verify that the suffix matches either the IP or control
                assay = sample_metadata["assay"].values[0]
                if assay == "ChIP":
                    ip_value = sample_metadata["ip"].values[0]
                    control_value = sample_metadata["control"].values[0]

                    # Check if suffix matches IP or control (case-insensitive)
                    if (
                        suffix.lower() == str(ip_value).lower()
                        or suffix.lower() == str(control_value).lower()
                    ):
                        metadata_sample_name = base_name
                    else:
                        logger.warning(
                            f"Sample '{sample_name}' suffix '{suffix}' doesn't match IP '{ip_value}' or control '{control_value}'"
                        )
                        sample_metadata = pd.DataFrame()  # Clear match
                else:
                    logger.info(
                        f"Matched '{sample_name}' to metadata entry '{base_name}'"
                    )
                    metadata_sample_name = base_name

        if sample_metadata.empty:
            logger.warning(f"Sample '{sample_name}' not found in metadata, skipping")
            continue

        assay = sample_metadata["assay"].values[0]
        if assay not in assay_groups:
            assay_groups[assay] = {
                "bam_files": [],
                "sample_id": [],
                "metadata_names": [],
            }

        assay_groups[assay]["bam_files"].append(bam_file)
        assay_groups[assay]["sample_id"].append(sample_name)
        assay_groups[assay]["metadata_names"].append(metadata_sample_name)
        sample_to_metadata[sample_name] = metadata_sample_name

    logger.info(f"Found {len(assay_groups)} assays: {list(assay_groups.keys())}")

    # Count total samples across all assays
    total_samples = sum(len(data["sample_id"]) for data in assay_groups.values())
    logger.info(f"Total samples across all assays: {total_samples}")
    for assay_name, assay_data in assay_groups.items():
        logger.info(f"  {assay_name}: {len(assay_data['sample_id'])} samples")

    # Process all samples in order (grouped by assay but flattened into single dimension)
    sparsity_values = []
    processed_samples = []
    sample_counter = 0

    for assay_idx, (assay_name, assay_data) in enumerate(assay_groups.items(), 1):
        for sample_idx, (bam_file, sample_name) in enumerate(
            zip(assay_data["bam_files"], assay_data["sample_id"]), 1
        ):
            sample_counter += 1
            seqnado_marker = "seqnado_output/"
            bam_file_short = str(bam_file)
            if seqnado_marker in bam_file_short:
                bam_file_short = bam_file_short.split(seqnado_marker, 1)[-1]
            logger.info(
                f"Processing [{sample_counter}/{total_samples}] {assay_name} sample {sample_name} from '{bam_file_short}'"
            )

            # Process all chromosomes for this sample
            sample_data = {}
            if max_workers > 1:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_contig = {
                        executor.submit(
                            process_chromosome, bam_file, contig, size
                        ): contig
                        for contig, size in chromsizes_dict.items()
                    }

                    for future in as_completed(future_to_contig):
                        contig = future_to_contig[future]
                        try:
                            contig_name, ds_chrom, sparsity = future.result()
                            sample_data[contig_name] = ds_chrom["signal"].values
                            sparsity_values.append(sparsity)
                        except Exception as e:
                            logger.error(
                                f"    Failed to process chromosome '{contig}': {e}"
                            )
                            raise
            else:
                # Sequential processing
                for contig, size in chromsizes_dict.items():
                    try:
                        contig_name, ds_chrom, sparsity = process_chromosome(
                            bam_file, contig, size
                        )
                        sample_data[contig_name] = ds_chrom["signal"].values
                        sparsity_values.append(sparsity)
                    except Exception as e:
                        logger.error(
                            f"    Failed to process chromosome '{contig}': {e}"
                        )
                        raise

            _append_sample_to_store(
                sample_data=sample_data,
                sample_name=sample_name,
                chromosomes=chromosomes,
                chromsizes_dict=chromsizes_dict,
                store_path=temp_store_path,
                is_first_sample=(sample_counter == 1),
            )
            processed_samples.append(sample_name)

    # Add metadata coordinates and global attributes
    logger.info("Adding metadata and global attributes...")
    ds_final = xr.open_zarr(temp_store_path, consolidated=False)

    # Add assay as a coordinate along the sample dimension
    assay_coord = []
    for sample_name in processed_samples:
        # Find which assay this sample belongs to
        for assay_name, assay_data in assay_groups.items():
            if sample_name in assay_data["sample_id"]:
                assay_coord.append(assay_name)
                break

    ds_final = ds_final.assign_coords(assay=("sample", assay_coord))

    # Add other metadata columns as coordinates
    for col in metadata.columns:
        if col not in [sample_col, "assay"]:
            metadata_values = []
            for sample_name in processed_samples:
                metadata_sample_name = sample_to_metadata[sample_name]
                sample_row = metadata[metadata[sample_col] == metadata_sample_name]
                if not sample_row.empty:
                    metadata_values.append(sample_row[col].values[0])
                else:
                    metadata_values.append(np.nan)
            ds_final = ds_final.assign_coords({col: ("sample", metadata_values)})

    # Add global attributes
    assay_names = list(assay_groups.keys())
    ds_final.attrs["description"] = (
        "BAM coverage data across all samples and chromosomes (flattened structure)"
    )
    ds_final.attrs["num_assays"] = len(assay_names)
    ds_final.attrs["num_samples_total"] = total_samples
    ds_final.attrs["num_chromosomes"] = len(chromosomes)
    ds_final.attrs["assays"] = ",".join(assay_names)
    ds_final.attrs["structure"] = "flattened (sample × chromosome × position)"

    ds_final.to_zarr(temp_store_path, mode="a", consolidated=False)

    # Optionally compress to ZipStore
    if use_zip and temp_store_path != final_store_path:
        logger.info(f"Compressing to ZipStore: {final_store_path}")
        _compress_to_zipstore(temp_store_path, final_store_path)
        logger.info(f"Removing temporary directory: {temp_store_path}")
        shutil.rmtree(temp_store_path)

    logger.info(
        f"Successfully created unified dataset with {total_samples} samples across {len(assay_names)} assays"
    )
    logger.info(f"Assays: {assay_names}")
    logger.info("Structure: flattened (sample × chromosome × position)")
    logger.info("No padding needed - each assay has its natural sample count")
    logger.info(f"Dataset saved to: {final_store_path if use_zip else temp_store_path}")


def bams_to_zarr(
    bam_files: list[str],
    chromsizes: str | Path | dict,
    store_path: Path,
    filter_chromosomes: bool = True,
    max_workers: int = 1,
    overwrite: bool = True,
    metadata: pd.DataFrame | Path | str | None = None,
    use_zip: bool = False,
    group_by_assay: bool = True,
) -> None:
    """
    Process multiple BAM files and combine them into a single unified Zarr dataset.

    This function creates a single xarray Dataset with dimensions:
    - If group_by_assay=True: (sample × chromosome × position)
      where 'assay' is stored as a coordinate along the sample dimension
      This avoids padding and efficiently handles different sample counts per assay
    - If group_by_assay=False: (sample × chromosome × position)

    Samples are written incrementally to reduce memory usage.

    Parameters:
    - bam_files: List of paths to BAM files.
    - chromsizes: Either a path to a chrom.sizes file or a dictionary mapping chromosome names to sizes.
    - store_path: Path to the main Zarr store.
    - filter_chromosomes: If True, only include main chromosomes (chr1-22, X, Y, M). Default: True.
    - max_workers: Number of parallel threads for chromosome processing within each BAM. Default: 1.
    - overwrite: If True, overwrite existing store. If False, skip existing samples (resume). Default: True.
    - metadata: Optional metadata as DataFrame, or path to CSV file with sample metadata.
                   Must have a column matching sample names and an 'assay' column.
    - use_zip: If True, compress to ZipStore after writing. Default: False (use DirectoryStore).
    - group_by_assay: If True, add assay as a dimension. Requires metadata with 'assay' column. Default: True.

    Returns:
    - None
    """
    # Determine paths
    if use_zip:
        # Use temporary directory, then compress to zip at end
        temp_store_path = Path(str(store_path).replace(".zarr", ".zarr.tmp"))
        final_store_path = store_path
    else:
        # Use directory store directly
        temp_store_path = store_path
        final_store_path = store_path

    # Check if store exists
    if temp_store_path.exists():
        if overwrite:
            logger.warning(f"Deleting existing Zarr store at: {temp_store_path}")
            if temp_store_path.is_file():
                temp_store_path.unlink()
            else:
                shutil.rmtree(temp_store_path)
        else:
            logger.info(f"Resuming from existing store: {temp_store_path}")
    chromsizes_dict = _parse_chromsizes(chromsizes, filter_chromosomes)

    # Load metadata if it's a path
    if metadata is not None and isinstance(metadata, (str, Path)):
        logger.info(f"Loading metadata from {metadata}")
        metadata = pd.read_csv(metadata)
        # Ensure standard dtypes for all columns
        for col in metadata.columns:
            values = metadata[col].values
            # Use np.str_ instead of np.unicode_
            if np.issubdtype(values.dtype, np.str_) or values.dtype == object:
                metadata[col] = metadata[col].astype(str)
            elif pd.api.types.is_integer_dtype(values):
                metadata[col] = metadata[col].astype(int)
            elif pd.api.types.is_float_dtype(values):
                metadata[col] = metadata[col].astype(float)

    # Validate metadata if grouping by assay
    if group_by_assay:
        if metadata is None:
            raise ValueError(
                "group_by_assay=True requires metadata with 'assay' column"
            )
        if "assay" not in metadata.columns:
            raise ValueError(
                "metadata must have 'assay' column when group_by_assay=True"
            )
        
    logger.info(
        f"Processing {len(bam_files)} BAM files into unified dataset: '{store_path}'"
    )

    # Extract sample names from BAM files
    sample_id = [Path(f).stem for f in bam_files]
    chromosomes = list(chromsizes_dict.keys())

    if group_by_assay:
        # Group samples by assay and process separately
        _process_assay(
            bam_files=bam_files,
            sample_id=sample_id,
            metadata=metadata,
            chromsizes_dict=chromsizes_dict,
            chromosomes=chromosomes,
            temp_store_path=temp_store_path,
            final_store_path=final_store_path,
            max_workers=max_workers,
            overwrite=overwrite,
            use_zip=use_zip,
        )
        return

    # Original processing without assay dimension
    # Initialize store with first sample to set up structure
    sparsity_values = []
    processed_samples = []

    for sample_idx, (bam_file, sample_name) in enumerate(zip(bam_files, sample_id), 1):
        # Check if sample already exists (for resume capability)
        if not overwrite and temp_store_path.exists():
            try:
                existing_ds = xr.open_zarr(temp_store_path, consolidated=False)
                if sample_name in existing_ds.sample.values:
                    logger.info(
                        f"[{sample_idx}/{len(bam_files)}] Skipping existing sample '{sample_name}'"
                    )
                    processed_samples.append(sample_name)
                    continue
            except Exception:
                pass  # Store doesn't exist or is invalid, continue processing

        # Process all chromosomes for this sample
        sample_data = {}
        if max_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_contig = {
                    executor.submit(process_chromosome, bam_file, contig, size): contig
                    for contig, size in chromsizes_dict.items()
                }

                for future in as_completed(future_to_contig):
                    contig = future_to_contig[future]
                    try:
                        contig_name, ds_chrom, sparsity = future.result()
                        sample_data[contig_name] = ds_chrom["signal"].values
                        sparsity_values.append(sparsity)
                        logger.info(f"  Completed chromosome '{contig_name}'")
                    except Exception as e:
                        logger.error(f"  Failed to process chromosome '{contig}': {e}")
                        raise
        else:
            # Sequential processing
            for contig, size in chromsizes_dict.items():
                try:
                    contig_name, ds_chrom, sparsity = process_chromosome(
                        bam_file, contig, size
                    )
                    sample_data[contig_name] = ds_chrom["signal"].values
                    sparsity_values.append(sparsity)
                    logger.info(f"  Completed chromosome '{contig_name}'")
                except Exception as e:
                    logger.error(f"  Failed to process chromosome '{contig}': {e}")
                    raise

        # Write this sample to the store incrementally
        logger.info(f"  Writing sample '{sample_name}' to Zarr store...")
        _append_sample_to_store(
            sample_data=sample_data,
            sample_name=sample_name,
            chromosomes=chromosomes,
            chromsizes_dict=chromsizes_dict,
            store_path=temp_store_path,
            is_first_sample=(sample_idx == 1 and not processed_samples),
        )
        processed_samples.append(sample_name)

    # Add metadata and finalize
    logger.info("Finalizing dataset with metadata...")
    _finalize_store(
        store_path=temp_store_path,
        sample_id=processed_samples,
        metadata=metadata,
        sparsity_values=sparsity_values,
        num_chromosomes=len(chromosomes),
    )

    # Optionally compress to ZipStore
    if use_zip and temp_store_path != final_store_path:
        logger.info(f"Compressing to ZipStore: {final_store_path}")
        _compress_to_zipstore(temp_store_path, final_store_path)
        logger.info(f"Removing temporary directory: {temp_store_path}")
        shutil.rmtree(temp_store_path)

    avg_sparsity = np.mean(sparsity_values) if sparsity_values else 0
    logger.info(
        f"Successfully created unified dataset with {len(processed_samples)} samples and {len(chromosomes)} chromosomes"
    )
    logger.info(f"Average sparsity: {avg_sparsity:.2f}%")
    logger.info(f"Dataset saved to: {final_store_path}")
    # Consolidate Zarr metadata for faster loading
    zarr.consolidate_metadata(str(final_store_path))
    logger.info(f"Consolidated Zarr metadata at: {final_store_path}")
