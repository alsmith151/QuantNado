import os
import warnings

os.environ["KMP_WARNINGS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", message="pkg_resources*")

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd
from loguru import logger


from QuantNado.call_quantile_peaks import call_peaks_from_bigwig_dir
from QuantNado.make_dataset import make_dataset
from QuantNado.make_zarr_store import combine_cached_zarrs, process_bam
from QuantNado.combine_metadata import combine_metadata_files, find_metadata_files


def call_peaks_main():
    parser = argparse.ArgumentParser(
        description="Call quantile-based peaks from bigWig files"
    )
    parser.add_argument(
        "--bigwig-dir",
        required=True,
        type=Path,
        help="Directory containing bigWig files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to save output peak files (BED format)",
    )
    parser.add_argument(
        "--chromsizes",
        required=True,
        help="Path to a two-column chromsizes file (chromosome, size)",
    )
    parser.add_argument(
        "--blacklist", default=None, help="Path to a BED file with regions to exclude"
    )
    parser.add_argument(
        "--tilesize",
        type=int,
        default=128,
        help="Size of genomic tiles to create (default: 128 bp)",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.98,
        help="Quantile threshold for peak calling (default: 0.98)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge overlapping peaks after quantile calling (default: False)",
    )
    parser.add_argument(
        "--tmp-dir",
        default="tmp",
        type=Path,
        help="Temporary directory for intermediate files (default: 'tmp')",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="quantnado_processing.log",
        help="Path to the log file (default: quantnado_processing.log)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.log_file.parent.exists():
        args.log_file.parent.mkdir(parents=True, exist_ok=True)

    if args.log_file.exists():
        args.log_file.unlink()

    _setup_logging(args.log_file, args.verbose)

    try:
        call_peaks_from_bigwig_dir(
            bigwig_dir=args.bigwig_dir,
            output_dir=args.output_dir,
            chromsizes_file=args.chromsizes,
            blacklist_file=args.blacklist,
            tilesize=args.tilesize,
            quantile=args.quantile,
            merge=args.merge,
            tmp_dir=args.tmp_dir,
        )
        logger.success(f"Finished calling peaks: {args.output_dir}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Peak calling failed: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def make_dataset_main():
    parser = argparse.ArgumentParser(
        description="Generate an AnnData or MuData dataset from bigWigs"
    )
    parser.add_argument(
        "--bigwig-dir",
        required=True,
        type=Path,
        help="Directory containing bigWig files",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        type=Path,
        help="Output file path for the dataset (AnnData with .h5ad extension)",
    )
    parser.add_argument(
        "--chromsizes",
        required=True,
        help="Path to a two-column chromsizes file (chromosome, size)",
    )
    parser.add_argument(
        "--regions", default=None, help="Path to a BED file with regions to use"
    )
    parser.add_argument(
        "--binsize",
        type=int,
        default=128,
        help="Size of genomic bins to create (if --regions is not provided)",
    )
    parser.add_argument(
        "--blacklist", default=None, help="Path to a BED file with regions to exclude"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="quantnado_processing.log",
        help="Path to the log file (default: quantnado_processing.log)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.log_file.parent.exists():
        args.log_file.parent.mkdir(parents=True, exist_ok=True)

    if args.log_file.exists():
        args.log_file.unlink()

    _setup_logging(args.log_file, args.verbose)

    try:
        make_dataset(
            bigwig_dir=args.bigwig_dir,
            output_file=args.output_file,
            chromsizes_file=args.chromsizes,
            regions_bed=args.regions,
            blacklist_file=args.blacklist,
            binsize=args.binsize,
        )
        logger.success(f"Finished building dataset: {args.output_file}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Dataset generation failed: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def make_zarr_main():
    parser = argparse.ArgumentParser(
        description="Process BAM files and create a Zarr dataset."
    )
    parser.add_argument(
        "--bam-file",
        help="Path to a single BAM file to process and cache.",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="Directory to store cached Zarr files.",
    )
    parser.add_argument(
        "--chromsizes",
        help="Path to a two-column chromsizes file (chromosome, size).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel threads for processing chromosomes (default: 4).",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine cached Zarr files into a final dataset.",
    )
    parser.add_argument(
        "--metadata-path",
        help="Path to metadata CSV file.",
    )
    parser.add_argument(
        "--output-path",
        help="Path to save the combined Zarr dataset.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("quantnado_processing.log"),
        help="Path to the log file (default: quantnado_processing.log).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.log_file.parent != Path(".") and not args.log_file.parent.exists():
        args.log_file.parent.mkdir(parents=True, exist_ok=True)

    # Don't delete the log file - append to it across multiple runs
    _setup_logging(args.log_file, args.verbose)

    if args.bam_file:
        if not args.chromsizes:
            raise ValueError("--chromsizes is required when processing BAM files.")

        logger.info(f"Processing {args.bam_file} for all chromosomes")
        process_bam(
            bam_file=args.bam_file,
            chromsizes=args.chromsizes,
            cache_dir=args.cache_dir,
            max_workers=args.max_workers,
        )

    if args.combine:
        if not args.metadata_path or not args.output_path:
            raise ValueError(
                "--metadata-path and --output-path are required for combining datasets."
            )

        metadata_df = pd.read_csv(args.metadata_path)
        combine_cached_zarrs(
            cache_dir=args.cache_dir,
            metadata_df=metadata_df,
            output_path=Path(args.output_path),
        )

    logger.success("Zarr processing complete.")


def combine_metadata_main():
    """CLI entry point for combining metadata files."""
    parser = argparse.ArgumentParser(
        description="Combine metadata CSV files from different assays."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing metadata_*.csv files to combine.",
    )
    parser.add_argument(
        "--metadata-files",
        nargs="+",
        type=Path,
        help="Specific metadata CSV files to combine (alternative to --data-dir).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to save the combined metadata CSV file.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("quantnado_metadata.log"),
        help="Path to the log file (default: quantnado_metadata.log).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.log_file.parent != Path(".") and not args.log_file.parent.exists():
        args.log_file.parent.mkdir(parents=True, exist_ok=True)

    if args.log_file.exists():
        args.log_file.unlink()

    _setup_logging(args.log_file, args.verbose)

    # Determine which metadata files to combine
    if args.data_dir:
        metadata_files = find_metadata_files(args.data_dir)
    elif args.metadata_files:
        metadata_files = args.metadata_files
    else:
        raise ValueError("Either --data-dir or --metadata-files must be provided.")

    # Combine metadata files
    combined_df = combine_metadata_files(
        metadata_files=metadata_files,
        output_path=args.output_path,
    )

    logger.info("Metadata combining complete.")
    print(f"Combined metadata saved to {args.output_path}")
    print(f"Total samples: {len(combined_df)}")


def _setup_logging(log_path: Path, verbose: bool):
    logger.remove()
    log_format = "{time:YYYY-MM-DD HH:mm:ss} [{level}] {message}"
    logger.add(log_path, level="DEBUG", format=log_format, mode="a")
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO", format=log_format)
