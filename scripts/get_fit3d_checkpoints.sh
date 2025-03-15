#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# This script downloads and extracts the FiT3D DINOv2 from Hugging Face.
#
# Usage example:
#   chmod +x scripts/get_fit3d_checkpoints.sh
#   ./scripts/get_fit3d_checkpoints.sh
#
# After running, you will have a new folder in:
#   ./fit3d_checkpoints
# ------------------------------------------------------------------------------

set -e  # Exit immediately if a command exits with a non-zero status

echo "Creating fit3d_checkpoints folder (if not exists)..."
mkdir -p fit3d_checkpoints

echo "Switching to fit3d_checkpoints folder..."
cd fit3d_checkpoints
echo "Now in: $(pwd)"

echo "Downloading archive files for dinov2_base_finetuned..."
wget https://huggingface.co/yuanwenyue/FiT3D/blob/main/dinov2_base_finetuned.pth

echo "Downloading archive files for dinov2_reg_small_finetuned..."
wget https://huggingface.co/yuanwenyue/FiT3D/blob/main/dinov2_reg_small_finetuned.pth

echo "Downloading archive files for dinov2_small_finetuned..."
wget https://huggingface.co/yuanwenyue/FiT3D/blob/main/dinov2_small_finetuned.pth

echo "DINOv2 checkpoints downloaded and placed successfully!"