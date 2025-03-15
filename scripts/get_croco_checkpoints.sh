#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# This script downloads and extracts the CroCov2 pretrained models.
#
# Usage example:
#   chmod +x scripts/get_croco_checkpoints.sh
#   ./scripts/get_croco_checkpoints.sh
#
# After running, you will have a new folder in:
#   ./croco_pretrained_models
# ------------------------------------------------------------------------------

set -e  # Exit immediately if a command exits with a non-zero status

echo "Creating croco_pretrained_models folder (if not exists)..."
mkdir -p croco_pretrained_models

echo "Switching to croco_pretrained_models folder..."
cd croco_pretrained_models
echo "Now in: $(pwd)"

echo "Downloading CroCo.pth..."
wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo.pth

echo "Downloading CroCo_V2_ViTBase_SmallDecoder.pth..."
wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTBase_SmallDecoder.pth

echo "Downloading CroCo_V2_ViTBase_BaseDecoder.pth..."
wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTBase_BaseDecoder.pth

echo "Downloading CroCo_V2_ViTLarge_BaseDecoder.pth..."
wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTLarge_BaseDecoder.pth

echo "CroCov2 pretrained models downloaded and placed successfully!"