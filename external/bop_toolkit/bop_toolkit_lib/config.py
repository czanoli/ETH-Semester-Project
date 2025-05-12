# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Configuration of the BOP Toolkit."""

import os


######## Basic ########

# Folder with the BOP datasets.
if "BOP_PATH" in os.environ:
    datasets_path = os.environ["BOP_PATH"]
else:
    datasets_path = r"/path/to/bop/datasets"

# Folder with pose results to be evaluated.
results_path = r"/home/tatiana/chris-sem-prj/ETH-Semester-Project/bop_datasets/inference/lmo_fit3d_layer11_upper_nopca"

# Folder for the calculated pose errors and performance scores.
eval_path = r"/home/tatiana/chris-sem-prj/ETH-Semester-Project/bop_datasets/inference/lmo_fit3d_layer11_upper_nopca"

######## Extended ########

# Folder for outputs (e.g. visualizations).
output_path = r"/home/tatiana/chris-sem-prj/ETH-Semester-Project/bop_datasets"

# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r"/path/to/bop_renderer/build"

# Executable of the MeshLab server.
meshlab_server_path = r"/path/to/meshlabserver.exe"

# Number of workers for the parallel evaluation of pose errors.
num_workers = 10

# use torch to calculate the errors
use_gpu = False
