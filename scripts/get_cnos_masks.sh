#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# This script downloads the CNOS-FastSAM segmentation mask archives
# and places them into the correct subfolder.
# Make sure you run this from your "foundpose" directory, or adjust paths as needed.
# ------------------------------------------------------------------------------

set -e

echo "Switching to 'bop_datasets' folder (should already contain your BOP data)..."
cd bop_datasets
echo "Now in: $(pwd)"

echo "Creating detections/cnos-fastsam folders (if not exists)..."
mkdir -p detections/cnos-fastsam

echo "Switching to detections/cnos-fastsam folder..."
cd detections/cnos-fastsam
echo "Now in: $(pwd)"

echo "Downloading CNOS-FastSAM detection masks..."
wget https://bop.felk.cvut.cz/media/data/bop_datasets_extra/bop23_default_detections_for_task4.zip

echo "Unzipping detection archives..."
unzip -q bop23_default_detections_for_task4.zip

echo "Moving cnos-fastsam data into current folder..."
mv bop23_default_detections_for_task4/cnos-fastsam/* .

echo "Removing leftover files and folders..."
rm -f bop23_default_detections_for_task4.zip
rm -rf bop23_default_detections_for_task4
rm -rf __MACOSX

echo "CNOS-FastSAM segmentation masks downloaded and placed successfully!"
