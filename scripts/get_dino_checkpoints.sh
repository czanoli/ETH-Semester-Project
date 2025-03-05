#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# This script downloads and extracts the FiT3D DINOv2 from Hugging Face.
#
# Usage example:
#   chmod +x get_dino_checkpoints.sh
#   ./get_dino_checkpoints
#
# After running, you will have a new folder in:
#   ./dino_checkpoints
# ------------------------------------------------------------------------------

set -e  # Exit immediately if a command exits with a non-zero status

echo "Creating dino_checkpoints folder (if not exists)..."
mkdir -p dino_checkpoints

echo "Switching to dino_checkpoints folder..."
cd dino_checkpoints
echo "Now in: $(pwd)"

echo "Downloading archive files for dinov2_base_finetuned..."
wget https://huggingface.co/yuanwenyue/FiT3D/blob/main/dinov2_base_finetuned.pth

echo "Downloading archive files for dinov2_reg_small_finetuned..."
wget https://huggingface.co/yuanwenyue/FiT3D/blob/main/dinov2_reg_small_finetuned.pth

echo "Downloading archive files for dinov2_small_finetuned..."
wget https://huggingface.co/yuanwenyue/FiT3D/blob/main/dinov2_small_finetuned.pth

echo "DINOv2 checkpoints downloaded and placed successfully!"