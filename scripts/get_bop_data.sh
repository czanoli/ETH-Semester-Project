#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# This script downloads and extracts the lmo BOP dataset from Hugging Face.
#
# Usage example (from project root folder):
#   chmod +x scripts/get_lmo_bop_data.sh
#   ./scripts/get_lmo_bop_data.sh
#
# After running, you will have a new folder in:
#   ./bop_datasets/lmo
# ------------------------------------------------------------------------------

set -e  # Exit immediately if a command exits with a non-zero status
DATASET_NAME = "lmo"
echo "Dataset name: ${DATASET_NAME}"

# Make a folder for all BOP datasets if it doesn't exist
echo "Creating 'bop_datasets' folder under current directory (if not exists)..."
mkdir -p bop_datasets

echo "Switching to bop_datasets folder..."
cd bop_datasets
echo "Now in: $(pwd)"

# Construct the download source URL
SRC="https://huggingface.co/datasets/bop-benchmark/${DATASET_NAME}/resolve/main"
echo "Download source: ${SRC}"

echo "Downloading archive files for '${DATASET_NAME}' dataset..."
wget "${SRC}/${DATASET_NAME}_base.zip"
wget "${SRC}/${DATASET_NAME}_models.zip"
wget "${SRC}/${DATASET_NAME}_test_bop19.zip"

echo "Unzipping ${DATASET_NAME}_base.zip..."
unzip -q "${DATASET_NAME}_base.zip"

echo "Unzipping ${DATASET_NAME}_models.zip to '${DATASET_NAME}' folder..."
unzip -q "${DATASET_NAME}_models.zip" -d "${DATASET_NAME}"

echo "Unzipping ${DATASET_NAME}_test_bop19.zip to '${DATASET_NAME}' folder..."
unzip -q "${DATASET_NAME}_test_bop19.zip" -d "${DATASET_NAME}"

echo "Removing leftover zip files..."
rm -f *.zip

echo "BOP '${DATASET_NAME}' dataset download and extraction complete!"
