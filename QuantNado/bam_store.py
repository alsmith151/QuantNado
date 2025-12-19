"""BamStore: A unified interface for storing BAM-derived coverage data."""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import bamnado
import dask.array as da
import numpy as np
import pandas as pd
import sparse
import xarray as xr
from loguru import logger


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def write_initial(
        self,
        data: xr.Dataset,
        encoding: dict,
        store_path: Path,
    ) -> None:
        """Write initial dataset to create the store."""
        pass

    @abstractmethod
    def append(
        self,
        data: xr.Dataset,
        store_path: Path,
        append_dim: str,
    ) -> None:
        """Append data along a dimension."""
        pass

    @abstractmethod
    def open(self, store_path: Path) -> xr.Dataset:
        """Open an existing store."""
        pass

    @abstractmethod
    def finalize(self, store_path: Path) -> None:
        """Finalize the store (e.g., consolidate metadata)."""
        pass

    @abstractmethod
    def exists(self, store_path: Path) -> bool:
        """Check if store exists."""
        pass

    @abstractmethod
    def delete(self, store_path: Path) -> None:
        """Delete the store."""
        pass


class ZarrBackend(StorageBackend):
    """Zarr storage backend."""

    def write_initial(
        self,
        data: xr.Dataset,
        encoding: dict,
        store_path: Path,
    ) -> None:
        """Write initial dataset to create the Zarr store."""
        data.to_zarr(
            store_path,
            mode="w",
            encoding=encoding,
            consolidated=False,
        )
        logger.debug(f"Created Zarr store at {store_path}")

    def append(
        self,
        data: xr.Dataset,
        store_path: Path,
        append_dim: str,
    ) -> None:
        """Append data to Zarr store along a dimension."""
        data.to_zarr(
            store_path,
            mode="a",
            append_dim=append_dim,
            consolidated=False,
        )

    def open(self, store_path: Path) -> xr.Dataset:
        """Open an existing Zarr store."""
        return xr.open_zarr(store_path, consolidated=False)

    def finalize(self, store_path: Path) -> None:
        """Consolidate Zarr metadata for faster loading."""
        import zarr

        zarr.consolidate_metadata(str(store_path))
        logger.info(f"Consolidated Zarr metadata at: {store_path}")

    def exists(self, store_path: Path) -> bool:
        """Check if Zarr store exists."""
        return store_path.exists() and (store_path / ".zgroup").exists()

    def delete(self, store_path: Path) -> None:
        """Delete the Zarr store."""
        import shutil

        if self.exists(store_path):
            shutil.rmtree(store_path)
            logger.debug(f"Deleted Zarr store at {store_path}")


class HDF5Backend(StorageBackend):
    """HDF5/NetCDF storage backend."""

    def write_initial(
        self,
        data: xr.Dataset,
        encoding: dict,
        store_path: Path,
    ) -> None:
        """Write initial dataset to create the HDF5 file."""
        data.to_netcdf(
            store_path,
            mode="w",
            encoding=encoding,
            format="NETCDF4",
        )
        logger.debug(f"Created HDF5 store at {store_path}")

    def append(
        self,
        data: xr.Dataset,
        store_path: Path,
        append_dim: str,
    ) -> None:
        """Append data to HDF5 file along a dimension."""
        data.to_netcdf(
            store_path,
            mode="a",
            append_dim=append_dim,
            format="NETCDF4",
        )

    def open(self, store_path: Path) -> xr.Dataset:
        """Open an existing HDF5 store."""
        return xr.open_dataset(store_path, engine="netcdf4")

    def finalize(self, store_path: Path) -> None:
        """Finalize HDF5 store (no-op for HDF5)."""
        logger.debug(f"HDF5 store finalized at: {store_path}")

    def exists(self, store_path: Path) -> bool:
        """Check if HDF5 store exists."""
        return store_path.exists() and store_path.is_file()

    def delete(self, store_path: Path) -> None:
        """Delete the HDF5 store."""
        if self.exists(store_path):
            store_path.unlink()
            logger.debug(f"Deleted HDF5 store at {store_path}")


BackendType = Literal["zarr", "hdf5", "netcdf4"]


class BamStore:
    """
    Unified interface for storing BAM-derived coverage data with support for multiple backends.

    Supports:
    - Zarr: Chunked array storage, ideal for large datasets and cloud storage
    - HDF5/NetCDF4: Traditional hierarchical format, good for compatibility

    Example:
        # Create a new store
        store = BamStore(store_path, backend="zarr", overwrite=True)
        store.write_sample(sample_data, sample_name, chromosomes, chromsizes_dict)
        store.finalize(metadata)

        # Open existing store
        store = BamStore.open(store_path, backend="zarr")
        ds = store.dataset
    """

    def __init__(
        self,
        store_path: Path | str,
        backend: BackendType = "zarr",
        overwrite: bool = True,
    ):
        """
        Initialize BamStore.

        Parameters:
        - store_path: Path to the store
        - backend: Storage backend to use ("zarr", "hdf5", "netcdf4")
        - overwrite: If True, delete existing store. If False, resume/append mode.
        """
        self.store_path = Path(store_path)
        self.backend_type = backend
        self._backend = self._get_backend(backend)
        self._is_initialized = False
        self._sample_count = 0

        # Handle existing store
        if self._backend.exists(self.store_path):
            if overwrite:
                logger.warning(f"Deleting existing store at: {self.store_path}")
                self._backend.delete(self.store_path)
            else:
                logger.info(f"Resuming with existing store: {self.store_path}")
                # Load existing dataset to get sample count
                try:
                    ds = self._backend.open(self.store_path)
                    self._sample_count = len(ds.sample)
                    self._is_initialized = True
                    logger.info(f"Found {self._sample_count} existing samples")
                except Exception as e:
                    logger.warning(f"Could not read existing store: {e}")

    @staticmethod
    def _get_backend(backend_type: BackendType) -> StorageBackend:
        """Get the appropriate storage backend."""
        if backend_type == "zarr":
            return ZarrBackend()
        elif backend_type in ("hdf5", "netcdf4"):
            return HDF5Backend()
        else:
            raise ValueError(
                f"Unknown backend: {backend_type}. "
                f"Supported backends: 'zarr', 'hdf5', 'netcdf4'"
            )

    @staticmethod
    def _parse_chromsizes(
        chromsizes: str | Path | dict[str, int], filter_chromosomes: bool = True
    ) -> dict[str, int]:
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

    @classmethod
    def open(
        cls,
        store_path: Path | str,
        backend: BackendType = "zarr",
    ) -> xr.Dataset:
        """
        Open an existing store and return the dataset.

        Parameters:
        - store_path: Path to the store
        - backend: Storage backend ("zarr", "hdf5", "netcdf4")

        Returns:
        - xarray Dataset
        """
        store_path = Path(store_path)
        backend_obj = cls._get_backend(backend)

        if not backend_obj.exists(store_path):
            raise FileNotFoundError(f"Store not found at: {store_path}")

        logger.info(f"Opening {backend} store at: {store_path}")
        return backend_obj.open(store_path)

    @staticmethod
    def _process_chromosome(
        bam_file: str, contig: str, contig_size: int
    ) -> tuple[str, xr.Dataset, float]:
        """
        Process a single chromosome from a BAM file.

        Parameters:
        - bam_file: Path to BAM file
        - contig: Chromosome name
        - contig_size: Chromosome size

        Returns:
        - contig: Chromosome name
        - ds_chrom: xarray Dataset with signal data
        - sparsity: Percentage of zero values
        """
        # Constants for bamnado
        BIN_SIZE = 1
        SCALE_FACTOR = 1.0
        USE_FRAGMENT = False

        # Get coverage signal using bamnado
        signal = bamnado.get_signal_for_chromosome(
            bam_path=bam_file,
            chromosome_name=contig,
            bin_size=BIN_SIZE,
            scale_factor=SCALE_FACTOR,
            use_fragment=USE_FRAGMENT,
            ignore_scaffold_chromosomes=False,
        )

        # Detect optimal dtype based on max value
        max_val = signal.max()
        if max_val <= np.iinfo(np.uint16).max:
            dtype = np.uint16
        elif max_val <= np.iinfo(np.uint32).max:
            dtype = np.uint32
        else:
            dtype = np.float32

        # Convert to selected dtype
        data = signal.astype(dtype)

        # Calculate sparsity
        sparsity = (np.sum(data == 0) / data.size) * 100

        # Create sparse array and xarray Dataset
        sparse_data = sparse.COO.from_numpy(data)
        ds_chrom = xr.Dataset(
            {"signal": (["position"], sparse_data)},
            coords={
                "position": np.arange(contig_size, dtype="int32"),
                "chromosome": contig,
            },
        )

        return contig, ds_chrom, sparsity

    def _process_bam_file(
        self,
        bam_file: str,
        chromsizes_dict: dict[str, int],
        max_workers: int = 1,
    ) -> tuple[dict[str, np.ndarray], list[float]]:
        """
        Process all chromosomes for a single BAM file in parallel.

        Parameters:
        - bam_file: Path to BAM file
        - chromsizes_dict: Dictionary mapping chromosome names to sizes
        - max_workers: Number of parallel threads for processing chromosomes

        Returns:
        - sample_data: Dictionary mapping chromosome names to signal arrays
        - sparsity_values: List of sparsity percentages for each chromosome
        """
        sample_data = {}
        sparsity_values = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_contig = {
                executor.submit(
                    self._process_chromosome, bam_file, contig, size
                ): contig
                for contig, size in chromsizes_dict.items()
            }

            for future in as_completed(future_to_contig):
                contig = future_to_contig[future]
                try:
                    contig_name, ds_chrom, sparsity = future.result()
                    # Extract sparse data properly
                    signal_data = ds_chrom["signal"].data
                    sample_data[contig_name] = signal_data
                    sparsity_values.append(sparsity)
                except Exception as e:
                    logger.error(f"  Failed to process chromosome '{contig}': {e}")
                    raise

        return sample_data, sparsity_values

    def write_sample(
        self,
        sample_data: dict[str, sparse.COO],
        sample_name: str,
        chromosomes: list[str],
        chromsizes_dict: dict[str, int],
    ) -> None:
        """
        Write a single sample to the store.

        Parameters:
        - sample_data: Dictionary mapping chromosome names to sparse coverage arrays
        - sample_name: Name of the sample
        - chromosomes: Ordered list of chromosome names
        - chromsizes_dict: Dictionary mapping chromosome names to sizes
        """
        # Determine max chromosome length for padding
        max_chrom_size = max(chromsizes_dict[chrom] for chrom in chromosomes)

        # Determine dtype from one chromosome (sparse arrays have dtype attribute)
        first_chrom = chromosomes[0]
        dtype = sample_data[first_chrom].dtype

        # Stack sparse arrays with padding - build list of padded sparse rows
        sparse_rows = []
        for chrom in chromosomes:
            chrom_data_sparse = sample_data[chrom]
            chrom_len = chromsizes_dict[chrom]

            # Pad sparse array to max_chrom_size if needed
            if chrom_len < max_chrom_size:
                # Create padded sparse array by converting to dense temporarily
                # (only for the padding step, individual chromosomes are small enough)
                padded = np.zeros(max_chrom_size, dtype=dtype)
                padded[:chrom_len] = chrom_data_sparse.todense()
                sparse_rows.append(sparse.COO.from_numpy(padded))
            else:
                sparse_rows.append(chrom_data_sparse)

        # Stack sparse arrays efficiently (creates 2D array: chromosomes × position)
        data_2d = sparse.stack(sparse_rows)

        # Wrap sparse array in dask for lazy evaluation and efficient chunking
        # Chunks: (1 chromosome, 1M positions) for efficient zarr storage
        dask_array = da.from_array(
            data_2d,
            chunks=(1, 1_000_000),
            asarray=False,  # Keep as sparse
        )

        # Add sample dimension by expanding to 3D (1 × chromosomes × position)
        dask_array_3d = dask_array[None, :, :]

        # Create Dataset with dask-wrapped sparse data
        ds_sample = xr.Dataset(
            {
                "signal": (
                    ["sample", "chromosome", "position"],
                    dask_array_3d,
                )
            },
            coords={
                "sample": np.array([str(sample_name)], dtype="object"),
                "chromosome": np.array([str(c) for c in chromosomes], dtype="object"),
                "position": np.arange(max_chrom_size, dtype="int32"),
            },
        )

        # Encoding for efficient storage
        encoding = {
            "signal": {
                "chunks": (1, 1, 1_000_000),
                "dtype": str(dtype),
            }
        }

        if not self._is_initialized:
            # First sample - create store
            self._backend.write_initial(ds_sample, encoding, self.store_path)
            self._is_initialized = True
            logger.debug(f"Initialized store with sample '{sample_name}'")
        else:
            # Append to existing store
            self._backend.append(ds_sample, self.store_path, append_dim="sample")
            logger.debug(f"Appended sample '{sample_name}'")

        self._sample_count += 1

    @staticmethod
    def _combine_metadata_files(metadata_files: list[Path | str]) -> pd.DataFrame:
        """
        Combine multiple metadata CSV files into a single DataFrame.

        Parameters:
        - metadata_files: List of paths to metadata CSV files

        Returns:
        - Combined DataFrame
        """
        if not metadata_files:
            raise ValueError("No metadata files provided")

        # Read all metadata files
        dataframes = []
        for file_path in metadata_files:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.warning(f"Metadata file not found: {file_path}, skipping")
                continue
            base_file_path = file_path.name
            logger.info(f"Reading metadata file: {base_file_path}")
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

        # Define preferred column order
        priority_cols = ["sample_id"]
        deprioritized_cols = sorted(
            [col for col in all_columns if "r1" in col.lower() or "r2" in col.lower()]
        )
        middle_cols = sorted(
            [
                col
                for col in all_columns
                if col not in priority_cols and col not in deprioritized_cols
            ]
        )
        all_columns_ordered = priority_cols + middle_cols + deprioritized_cols

        # Ensure all dataframes have the same column order
        dataframes = [df[all_columns_ordered] for df in dataframes]

        # Concatenate all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)

        return combined_df

    @classmethod
    def from_bam_files(
        cls,
        bam_files: list[str],
        chromsizes: str | Path | dict[str, int],
        store_path: Path | str,
        metadata: pd.DataFrame | Path | str | list[Path | str],
        filter_chromosomes: bool = True,
        max_workers: int = 1,
        overwrite: bool = True,
        sample_column: str = "sample_id",
        backend: BackendType = "zarr",
        log_file: Path | None = None,
    ) -> "BamStore":
        """
        Create a BamStore from BAM files with metadata.

        This is the main entry point for creating a store from BAM files. It handles:
        - Processing BAM files in parallel
        - Grouping by assay
        - Fuzzy matching sample names with metadata
        - Writing samples incrementally
        - Adding metadata coordinates

        Parameters:
        - bam_files: List of paths to BAM files
        - chromsizes: Either a path to a chrom.sizes file or a dictionary mapping chromosome names to sizes
        - store_path: Path to the store
        - metadata: DataFrame with sample metadata, path to CSV file, or list of paths to CSV files to combine
        - filter_chromosomes: If True, only include main chromosomes (chr1-22, X, Y, M). Default: True.
        - max_workers: Number of parallel threads for chromosome processing
        - overwrite: If True, overwrite existing store
        - sample_column: Column name in metadata containing sample identifiers
        - backend: Storage backend ('zarr', 'hdf5', 'netcdf4')
        - log_file: Optional path to log file. If None, logging is not configured.

        Returns:
        - BamStore instance
        """
        # Set up logging if log_file is provided
        if log_file is not None:
            from QuantNado.utils import setup_logging

            setup_logging(Path(log_file), verbose=False)

        # Load and combine metadata
        if isinstance(metadata, list):
            # Multiple metadata files - combine them
            logger.info(f"Combining {len(metadata)} metadata files")
            metadata = cls._combine_metadata_files(metadata)
        elif isinstance(metadata, (str, Path)):
            # Single metadata file
            logger.info(f"Loading metadata from {metadata}")
            metadata = pd.read_csv(metadata)
        # else: metadata is already a DataFrame

        # Ensure standard dtypes for all columns
        for col in metadata.columns:
            values = metadata[col].values
            if np.issubdtype(values.dtype, np.str_) or values.dtype == object:
                metadata[col] = metadata[col].astype(str)
            elif pd.api.types.is_integer_dtype(values):
                metadata[col] = metadata[col].astype(int)
            elif pd.api.types.is_float_dtype(values):
                metadata[col] = metadata[col].astype(float)

        logger.info(
            f"Processing {len(bam_files)} BAM files into unified dataset: '{store_path}'"
        )

        # Parse chromsizes
        chromsizes_dict = cls._parse_chromsizes(chromsizes, filter_chromosomes)

        # Initialize store
        store = cls(store_path, backend=backend, overwrite=overwrite)

        # Validate metadata
        if sample_column not in metadata.columns:
            raise ValueError(
                f"Sample column '{sample_column}' not found in metadata. "
                f"Available columns: {list(metadata.columns)}"
            )
        if "assay" not in metadata.columns:
            raise ValueError(
                f"Metadata must have 'assay' column. "
                f"Available columns: {list(metadata.columns)}"
            )

        # Extract sample names from BAM files
        sample_id = [Path(f).stem for f in bam_files]
        chromosomes = list(chromsizes_dict.keys())

        # Group samples by assay and track metadata name mapping
        assay_groups = {}
        sample_to_metadata = {}  # Maps BAM sample name to metadata sample name

        for bam_file, sample_name in zip(bam_files, sample_id):
            # Try exact match first
            sample_metadata = metadata[metadata[sample_column] == sample_name]

            # If not found, try fuzzy matching by removing suffix after last underscore
            metadata_sample_name = sample_name
            if sample_metadata.empty and "_" in sample_name:
                base_name = sample_name.rsplit("_", 1)[0]
                suffix = sample_name.rsplit("_", 1)[1]
                sample_metadata = metadata[metadata[sample_column] == base_name]

                if not sample_metadata.empty:
                    # For ChIP samples, verify suffix matches either IP or control
                    assay = sample_metadata["assay"].values[0]
                    if assay == "ChIP" or assay == "CUT&Tag":
                        ip_value = sample_metadata["ip"].values[0]
                        control_value = sample_metadata["control"].values[0]

                        if (
                            suffix.lower() == str(ip_value).lower()
                            or suffix.lower() == str(control_value).lower()
                        ):
                            metadata_sample_name = base_name
                        else:
                            logger.warning(
                                f"Sample '{sample_name}' suffix '{suffix}' doesn't match IP '{ip_value}' or control '{control_value}'"
                            )
                            sample_metadata = pd.DataFrame()
                    else:
                        logger.info(
                            f"Matched '{sample_name}' to metadata entry '{base_name}'"
                        )
                        metadata_sample_name = base_name

            if sample_metadata.empty:
                logger.warning(
                    f"Sample '{sample_name}' not found in metadata, skipping"
                )
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

        # Count total samples
        total_samples = sum(len(data["sample_id"]) for data in assay_groups.values())
        logger.info(f"Total samples across all assays: {total_samples}")
        for assay_name, assay_data in assay_groups.items():
            logger.info(f"  {assay_name}: {len(assay_data['sample_id'])} samples")

        # Process all samples
        sparsity_values = []
        processed_samples = []
        sample_counter = 0

        for assay_name, assay_data in assay_groups.items():
            for bam_file, sample_name in zip(
                assay_data["bam_files"], assay_data["sample_id"]
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
                sample_data, sample_sparsity = store._process_bam_file(
                    bam_file=bam_file,
                    chromsizes_dict=chromsizes_dict,
                    max_workers=max_workers,
                )
                sparsity_values.extend(sample_sparsity)

                # Write sample to store
                store.write_sample(
                    sample_data=sample_data,
                    sample_name=sample_name,
                    chromosomes=chromosomes,
                    chromsizes_dict=chromsizes_dict,
                )
                processed_samples.append(sample_name)

        # Add metadata coordinates and global attributes
        logger.info("Adding metadata and global attributes...")
        ds_final = store.dataset

        # Add assay as a coordinate along the sample dimension
        assay_coord = []
        for sample_name in processed_samples:
            for assay_name, assay_data in assay_groups.items():
                if sample_name in assay_data["sample_id"]:
                    assay_coord.append(assay_name)
                    break

        ds_final = ds_final.assign_coords(assay=("sample", assay_coord))

        # Add other metadata columns as coordinates
        for col in metadata.columns:
            if col not in [sample_column, "assay"]:
                metadata_values = []
                for sample_name in processed_samples:
                    metadata_sample_name = sample_to_metadata[sample_name]
                    sample_row = metadata[
                        metadata[sample_column] == metadata_sample_name
                    ]
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

        ds_final.to_zarr(store.store_path, mode="a", consolidated=False)

        # Finalize the store
        store.finalize(sparsity_values=sparsity_values)

        logger.info(
            f"Successfully created unified dataset with {total_samples} samples across {len(assay_names)} assays"
        )
        logger.info(f"Assays: {assay_names}")
        logger.info("Structure: flattened (sample × chromosome × position)")

        return store

    def add_metadata(
        self,
        metadata: pd.DataFrame,
        sample_id: list[str],
        sample_column: str = "sample_id",
    ) -> None:
        """
        Add metadata coordinates and attributes to the store.

        Parameters:
        - metadata: DataFrame with sample metadata
        - sample_id: Ordered list of sample names in the store
        - sample_column: Column name in metadata containing sample identifiers
        """
        logger.info("Adding metadata coordinates and attributes...")

        # Open the store
        ds = self._backend.open(self.store_path)

        # Reorder metadata to match sample order in dataset
        sample_col = sample_column
        if sample_col not in metadata.columns:
            raise ValueError(
                f"Sample column '{sample_col}' not found in metadata. "
                f"Available columns: {list(metadata.columns)}"
            )

        metadata = metadata.set_index(sample_col).loc[sample_id].reset_index()

        # Add metadata columns as coordinates
        for col in metadata.columns:
            if col != sample_col:
                values = metadata[col].values
                # Convert to standard dtypes
                if values.dtype == object or np.issubdtype(values.dtype, np.str_):
                    values = values.astype(str)
                elif pd.api.types.is_integer_dtype(values):
                    values = values.astype(int)
                elif pd.api.types.is_float_dtype(values):
                    values = values.astype(float)
                ds = ds.assign_coords({col: ("sample", values)})

        # Add global attributes
        ds.attrs["num_samples"] = len(sample_id)
        ds.attrs["num_chromosomes"] = len(ds.chromosome)

        # Write back to store
        ds.to_zarr(self.store_path, mode="a", consolidated=False)
        logger.info("Metadata added successfully")

    def finalize(self, sparsity_values: list[float] | None = None) -> None:
        """
        Finalize the store (consolidate metadata, add final attributes).

        Parameters:
        - sparsity_values: Optional list of sparsity percentages
        """
        if sparsity_values:
            avg_sparsity = np.mean(sparsity_values)
            logger.info(f"Average sparsity: {avg_sparsity:.2f}%")

            # Add sparsity to attributes
            ds = self._backend.open(self.store_path)
            ds.attrs["average_sparsity"] = f"{avg_sparsity:.2f}%"
            ds.to_zarr(self.store_path, mode="a", consolidated=False)

        # Backend-specific finalization
        self._backend.finalize(self.store_path)
        logger.info(f"Store finalized at: {self.store_path}")

    @property
    def dataset(self) -> xr.Dataset:
        """Get the dataset from the store."""
        return self._backend.open(self.store_path)

    @property
    def sample_count(self) -> int:
        """Get the current number of samples in the store."""
        return self._sample_count
