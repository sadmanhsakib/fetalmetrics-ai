"""
dataset.py
==========
Kaggle dataset acquisition and re-upload utilities for the HC18 Grand Challenge.

Responsibilities
----------------
* ``fetch_dataset``   — Download the HC18 dataset via KaggleHub and stage it
                        under ``data/raw/`` relative to the project root.
* ``upload_dataset``  — Push a locally processed dataset back to Kaggle as a
                        new (or updated) dataset version.

These helpers are intentionally separate from the preprocessing pipeline so
that acquisition and data-preparation steps can be run and audited
independently.
"""

from __future__ import annotations

import os
import shutil

import kagglehub

# Relative path used as the local staging root for downloaded datasets.
# All further processing scripts expect the raw data at ``data/raw/``.
_DESTINATION_PATH = "data/"


def fetch_dataset() -> None:
    """Download the HC18 Grand Challenge dataset and stage it at ``data/raw/``.

    The destination directory is wiped before each download to guarantee a
    reproducible, clean state regardless of any prior partial downloads.

    Notes
    -----
    Requires a valid ``~/.kaggle/kaggle.json`` credential file.  The KaggleHub
    cache is bypassed by removing the destination directory first; the freshly
    downloaded archive is then renamed from its versioned slug to the canonical
    ``raw/`` subdirectory.
    """
    # Remove the existing staging directory to ensure a clean, reproducible state.
    if os.path.exists(_DESTINATION_PATH):
        shutil.rmtree(_DESTINATION_PATH)

    os.makedirs(_DESTINATION_PATH, exist_ok=True)

    path = kagglehub.dataset_download("thanhbnhphan/hc18-grand-challenge")

    # Relocate the versioned download to the canonical data/raw/ path.
    shutil.move(path, _DESTINATION_PATH)
    os.rename(
        os.path.join(_DESTINATION_PATH, os.path.basename(path)),
        os.path.join(_DESTINATION_PATH, "raw"),
    )

    print(f"Dataset staged at {_DESTINATION_PATH}raw")


def upload_dataset(handle: str, local_dataset_dir: str, is_new: bool = False) -> None:
    """Publish a local directory to Kaggle as a dataset version.

    Parameters
    ----------
    handle:
        Kaggle dataset slug in ``<owner>/<dataset-name>`` format,
        e.g. ``"sadmanhsakib/hc18-processed-dataset"``.
    local_dataset_dir:
        Path to the directory that contains the files to publish.
    is_new:
        If ``True``, create the dataset for the first time (no version notes
        are required).  If ``False``, prompt for release notes before pushing
        an incremental update.
    """
    if is_new:
        kagglehub.dataset_upload(handle, local_dataset_dir)
    else:
        version_notes = input("Version notes: ")
        kagglehub.dataset_upload(handle, local_dataset_dir, version_notes=version_notes)

    print(f"Dataset uploaded to {handle}.")


if __name__ == "__main__":
    fetch_dataset()

    """
    upload_dataset(
        handle="sadmanhsakib/hc18-processed-dataset",
        local_dataset_dir="data/processed",
    )
    """
