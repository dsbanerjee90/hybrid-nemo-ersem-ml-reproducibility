#!/usr/bin/env python3
"""
Add HYB-SF provenance and copyright metadata to NetCDF global attributes.

This modifies NetCDF files in place.
"""

from pathlib import Path
from datetime import datetime, timezone
import argparse

from netCDF4 import Dataset


ATTRS = {
    "creator_name": "Deep S. Banerjee",
    "creator_institution": "Plymouth Marine Laboratory, United Kingdom",
    "created": "November 2025",
    "project": "HYB-SF: hybrid scale-factor machine-learning correction framework for coupled NEMO-FABM-ERSEM biogeochemical prediction",
    "associated_manuscript": "Banerjee et al. (2026), Online machine-learning corrections improve coupled ocean biogeochemical prediction",
    "processing_note": (
        "Processed/extracted/regridded input data prepared for the HYB-SF hybrid modelling "
        "workflow. Source datasets and processing context are described in Banerjee et al. (2026)."
    ),
    "source_acknowledgement": (
        "These files are derived or processed from the respective source datasets described in "
        "Banerjee et al. (2026). Original source-data ownership, licences and citation requirements "
        "remain with the original data providers."
    ),
    "copyright": (
        "Copyright (c) 2025-2026 Deep S. Banerjee, Plymouth Marine Laboratory, and co-authors "
        "for the processed HYB-SF data compilation, processing workflow and associated metadata, "
        "unless otherwise stated."
    ),
    "rights": (
        "Provided for scholarly review and reproducibility of the associated HYB-SF hybrid modelling "
        "manuscript. Redistribution, modification or reuse outside this purpose may require permission "
        "from the authors, Plymouth Marine Laboratory and/or the original data providers."
    ),
}


def append_history(ds: Dataset, message: str) -> None:
    old_history = getattr(ds, "history", "")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new_entry = f"{timestamp}: {message}"

    if old_history:
        ds.setncattr("history", f"{old_history}\n{new_entry}")
    else:
        ds.setncattr("history", new_entry)


def update_file(nc_file: Path, dry_run: bool = False) -> bool:
    if dry_run:
        print(f"[dry-run] would update: {nc_file}")
        return True

    try:
        with Dataset(nc_file, "a") as ds:
            for key, value in ATTRS.items():
                ds.setncattr(key, value)

            ds.setncattr(
                "metadata_updated",
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )

            append_history(
                ds,
                "Added HYB-SF provenance, copyright and manuscript-reference global attributes.",
            )

        print(f"[ok] {nc_file}")
        return True

    except Exception as exc:
        print(f"[failed] {nc_file}: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Add HYB-SF provenance/copyright global attributes to NetCDF files."
    )
    parser.add_argument(
        "--root",
        default="data/raw_external",
        help="Root directory containing HYB-SF NetCDF input files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be updated without modifying them.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    files = sorted(root.rglob("*.nc"))

    print(f"[info] root: {root}")
    print(f"[info] NetCDF files found: {len(files)}")

    if not files:
        raise SystemExit("No NetCDF files found. Check --root.")

    n_ok = 0
    n_fail = 0

    for nc_file in files:
        ok = update_file(nc_file, dry_run=args.dry_run)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print("\nSummary")
    print("-------")
    print(f"Updated/checked: {n_ok}")
    print(f"Failed:          {n_fail}")

    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
