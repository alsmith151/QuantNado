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
from QuantNado.make_zarr_store import bams_to_zarr
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
        description="Process BAM files and create a unified Zarr dataset."
    )
    parser.add_argument(
        "--bam-files",
        nargs="+",
        required=True,
        help="Paths to BAM files to process.",
    )
    parser.add_argument(
        "--chromsizes",
        required=True,
        help="Path to a two-column chromsizes file (chromosome, size).",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        type=Path,
        help="Path to save the unified Zarr dataset.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        help="Path to metadata CSV file (optional).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel threads for processing chromosomes (default: 4).",
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

    _setup_logging(args.log_file, args.verbose)

    # Load metadata if provided
    metadata_df = None
    if args.metadata_path:
        logger.info(f"Loading metadata from {args.metadata_path}")
        metadata_df = pd.read_csv(args.metadata_path)

    # Process all BAM files into unified Zarr dataset
    logger.info(f"Processing {len(args.bam_files)} BAM files into {args.output_path}")
    bams_to_zarr(
        bam_files=args.bam_files,
        chromsizes=args.chromsizes,
        main_store_path=args.output_path,
        max_workers=args.max_workers,
        metadata_df=metadata_df,
    )

    logger.success(f"Zarr dataset created: {args.output_path}")


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
