import os
import json

# Get REPO_PATH from environment variable
repo_path = os.getenv("REPO_PATH")
if not repo_path:
    raise ValueError("Environment variable REPO_PATH is not set.")

# Define parent directory
parent_dir = os.path.join(repo_path, "bop_datasets/templates/v1/lmo")

new_path = os.path.join(repo_path, "bop_datasets")

# Check if the directory exists before proceeding
if not os.path.isdir(parent_dir):
    raise FileNotFoundError(f"Error: Directory {parent_dir} does not exist. Check REPO_PATH and folder structure.")

# Iterate through folders and modify metadata.json
for folder_name in os.listdir(parent_dir):
    folder_path = os.path.join(parent_dir, folder_name)
    metadata_file = os.path.join(folder_path, "metadata.json")

    if os.path.isfile(metadata_file):
        print(f"Processing: {metadata_file}")
        with open(metadata_file, "r") as f:
            data = json.load(f)

        # Update paths
        for entry in data:
            entry["rgb_image_path"] = entry["rgb_image_path"].replace(
                "/Users/evinpinar/Documents/Workspace/foundpose_output", new_path
            )
            entry["depth_map_path"] = entry["depth_map_path"].replace(
                "/Users/evinpinar/Documents/Workspace/foundpose_output", new_path
            )
            entry["binary_mask_path"] = entry["binary_mask_path"].replace(
                "/Users/evinpinar/Documents/Workspace/foundpose_output", new_path
            )

        # Save modified metadata
        with open(metadata_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Updated: {metadata_file}")

print("All metadata.json files updated successfully.")
