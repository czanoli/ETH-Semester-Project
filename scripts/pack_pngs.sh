#!/usr/bin/env bash

# Usage:
#   ./pack_pngs.sh /path/to/main_folder my_images

# First argument: path to the main folder
MAIN_FOLDER="$1"

# Second argument: desired name for the zip file (no .zip extension)
ZIP_NAME="$2"

# If either argument is missing, show usage and exit
if [ -z "$MAIN_FOLDER" ] || [ -z "$ZIP_NAME" ]; then
  echo "Usage: $0 /path/to/main_folder zip_file_name"
  exit 1
fi

# Create the zip archive, including only PNG files
zip -r "${ZIP_NAME}.zip" "$MAIN_FOLDER" -i "*.png"