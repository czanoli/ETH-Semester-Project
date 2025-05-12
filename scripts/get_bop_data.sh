#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# This script downloads and extracts the chosen BOP dataset from Hugging Face.
#
# Usage example (from project root folder):
#   chmod +x scripts/get_bop_data.sh
#   ./scripts/get_bop_data.sh
#
# After running, for example, you will have a new folder in:
#   ./bop_datasets/lmo
# ------------------------------------------------------------------------------

set -e  # Exit immediately if a command exits with a non-zero status
DATASET_NAME="tless"
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

if [ "${DATASET_NAME}" == "tless" ]; then
    TEST_ZIP_NAME="tless_test_primesense_bop19.zip"
    TEST_ZIP_URL="https://huggingface.co/datasets/bop-benchmark/tless/resolve/main/${TEST_ZIP_NAME}"
else
    TEST_ZIP_NAME="${DATASET_NAME}_test_bop19.zip"
    TEST_ZIP_URL="${SRC}/${TEST_ZIP_NAME}"
fi

wget "${TEST_ZIP_URL}"

echo "Unzipping ${DATASET_NAME}_base.zip..."
unzip -q "${DATASET_NAME}_base.zip"

echo "Unzipping ${DATASET_NAME}_models.zip to '${DATASET_NAME}' folder..."
unzip -q "${DATASET_NAME}_models.zip" -d "${DATASET_NAME}"

echo "Unzipping ${TEST_ZIP_NAME} to '${DATASET_NAME}' folder..."
unzip -q "${TEST_ZIP_NAME}" -d "${DATASET_NAME}"

echo "Removing leftover zip files..."
rm -f *.zip

echo "BOP '${DATASET_NAME}' dataset download and extraction complete!"

