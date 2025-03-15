This guide is for setting up the necessary environment and infrastructure to run the "Using FoundPose" commands.
 - `ssh <remote-machine>`
 - `/opt/TurboVNC/bin/vncserver` --> to run a new VNC server
 - (on your local machine): open TurboVNC Viewer app --> make sure to have `<remote-machine-IP><vnc-display-ID>`, e.g. 100.66.235.1:1 in the VNC server field of the GUI popup which appears on your local machine. 
- Insert the password. The xfce desktop environment starts on the remote host.

### 1. Generating templates <a name="render-the-templates"></a>
Now we need to setup the environment to being able to run the first foundpose commands, which generates the templates from a given BOP dataset:
(on the remote machine):
- Make the get_bop_data.sh script executable by running: `chmod +x scripts/get_bop_data.sh`
- From the `root` directory, run `./scripts/get_bop_data.sh <dataset-acronym>` to get the BOP dataset of interest. Valid `<dataset-acronym>` are: `hot3d`, `hope`, `handal`, `ipd`, `xyzibd`, `itodd`, `itoddmv`, `lm`, `lmo`, `ycbv`, `ruapc`, `tless`, `hb`, `icbin`, `icmi`, `tudl`, `tyol`.
- Update the ```output_path``` in the BOP config file located at ```external/bop_toolkit/bop_toolkit_lib/config.py```  to point to the root directory of your BOP datasets.
- Make the `get_cnos_masks.sh` script executable by running: `chmod +x scripts/get_cnos_masks.sh`
- From the `root` directory, run `./scripts/get_cnos_masks.sh` to get the default segmentations created for Task 4 at BOP 2023 Challenge.

Your BOP datasets should be organized in the following directory structure:
```bash
bop_datasets/               # This is your $BOP_PATH
├── lmo/                    # Dataset directory for LM-O
│   ├── camera.json
│   ├── dataset_info.md
│   ├── models/             # 3D models of the objects
│   ├── models_eval/        # Simplified models for evaluation
│   ├── test/               # Test images and annotations
│   └── ...
├── tudl/ 
├── ...
├── detections/
│   └── cnos-fastsam/
│       ├── cnos-fastsam_lmo_test.json
│       ├── cnos-fastsam_tudl_test.json
│       └── ...
└── ...
```

- Create the first environment `foundpose_cpu` by running:
`conda env create -f conda_foundpose_cpu.yaml`

Next, create the conda environment activation script to set the necessary environment variables. This script will run automatically when you activate the environment. The activation script is typically located at ```$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh```. You can find ```$CONDA_PREFIX``` by running:
```bash
conda info --envs
```
If the ```env_vars.sh``` file does not exist, create it. 

Edit the ```env_vars.sh``` file as follows:

```bash
#!/bin/sh

export REPO_PATH=/path/to/project/repository  # Replace with the path to the project (root).
export BOP_PATH=/path/to/bop/datasets  # Replace with the path to BOP datasets created before.

export PYTHONPATH=$REPO_PATH:$REPO_PATH/external/bop_toolkit:$REPO_PATH/external/dinov2
```

Finally, activate the conda environment. This environment will be used to run the first foundpose command, to generate object templates:
```bash
conda activate foundpose_cpu
```

You can render the templates using the following script and the provided configuration file (e.g., for the LM-O dataset). To use other datasets, create a similar configuration file accordingly. 

Run the following command:

```bash
python scripts/gen_templates.py --opts-path configs/gen_templates/lmo.json
```
This script generates images, masks, depth maps, and camera parameters in the $output_path.

### 2. Generating object representation <a name="create-object-representation"></a>
- If you are done with generating templates, deactivate the `foundpose_cpu` environment by running: `conda deactivate`
- Now create the `foundpose_gpu` environment by running:
`conda env create -f conda_foundpose_gpu.yaml`

Next, create the conda environment activation script to set the necessary environment variables. This script will run automatically when you activate the environment. The activation script is typically located at ```$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh```. You can find ```$CONDA_PREFIX``` by running:
```bash
conda info --envs
```
If the ```env_vars.sh``` file does not exist, create it. 

Edit the ```env_vars.sh``` file as follows:

```bash
#!/bin/sh

export REPO_PATH=/path/to/project/repository  # Replace with the path to the project (root).
export BOP_PATH=/path/to/bop/datasets  # Replace with the path to BOP datasets created before.

export PYTHONPATH=$REPO_PATH:$REPO_PATH/external/bop_toolkit:$REPO_PATH/external/dinov2
```
- Activate it by running `conda activate foundpose_gpu`

Now you can create the object representation using the following script and configuration file (e.g., for the LM-O dataset):
```
python scripts/gen_repre.py --opts-path configs/gen_repre/lmo.json
```

### 3. Inference <a name="run-pose-estimation"></a>
Now, using again `foundpose_gpu` environment, you the can run coarse-level pose estimation for the LM-O dataset using the following script and configuration file:  
```
python scripts/infer.py --opts-path configs/infer/lmo.json
```
This will generate output poses in the BOP format.

### 4. Evaluation <a name="pose-evaluation"></a>
Open the `prepare_bop_submission.py` and modify the following fields to match the following data based on the evaluation you need to run. For example:

```python
object_dataset = "lmo"
version = "fit3d"
object_lids = [1, 5, 6, 8, 9, 10, 11, 12]
```

Run `python scripts/prepare_bop_submission.py` and this will save a `.csv` file into `/bop_datasets/inference/<dataset_method>` (e.g. `/bop_datasets/inference/lmo_fit3d`). This `.csv` file is needed for the automatic computation of the BOP challenge evaluation metric, i.e. AR = (AR_VSD + AR_MSSD + AR_MSPD)/3. In the following command replace `NAME_OF_CSV_WITH_RESULTS` with the name of the generated `.csv` file, for example: `coarse_lmo-test.csv`.

In `external/bop_toolkit/bop_toolkit_lib/config.py` modify the following two lines sch that the last part of the path is referred to the method for which you want to evaluate the poses. For example here we have `lmo_fit3d`, but coould be `lmo_v1`, `lmo_crocov2`, etc...

```python

# Folder with pose results to be evaluated.
results_path = r"/home/tatiana/chris-sem-prj/ETH-Semester-Project/bop_datasets/inference/lmo_fit3d/"

# Folder for the calculated pose errors and performance scores.
eval_path = r"/home/tatiana/chris-sem-prj/ETH-Semester-Project/bop_datasets/inference/lmo_fit3d/"
```

Note: perform the following command in the terminal of the xfce desktop, because you need and EGL renderer for making it running, so if you run the following command in a terminal that is not inside the xfce desktop created using VNC server, it will complain with the folllwing error:

```bash
RuntimeError: Could not import backend "EGL":
...
FileNotFoundError: [Errno 2] No such file or directory: '/home/tatiana/chris-sem-prj/foundpose/bop_datasets/inference/lmo_v1/tmp1742033686/worker_0.json'
Traceback (most recent call last):
  File "/home/tatiana/chris-sem-prj/foundpose/external/bop_toolkit/scripts/eval_bop19_pose.py", line 186, in <module>
    raise RuntimeError("Calculation of pose errors failed.")
RuntimeError: Calculation of pose errors failed.
```

So, with the foundpose_gpu conda env activated and inside the terminal of the remote desktop env, you can now run: 
```
python external/bop_toolkit/scripts/eval_bop19_pose.py --renderer_type=vispy --result_filenames=coarse_lmo-test.csv
```

Note: It's important to have the foundpose_gpu conda env activated otherwise you will get some Module Not Found errors (you need the env_vars.sh you modified before)


### CroCov2 Checkpoints <a name="croco-checkpoints"></a>
Download the CroCov2 checkpoints:
- `chmod +x scripts/get_croco_checkpoints.sh`
- `./scripts/get_croco_checkpoints.sh`

The CroCov2 checkpoints will be saved into the `croco_pretrained_models` folder.


## [OLD:]
### DINOv2 Checkpoints <a name="dino-checkpoints"></a>
Download the DINOv2 checkpoints from 3DFiT:
- `chmod +x scripts/get_dino_checkpoints.sh`
- `./scripts/get_dino_checkpoints.sh`

The DINOv2 checkpoints from [FiT3D](https://huggingface.co/yuanwenyue/FiT3D/tree/main) will be saved into the `dino_checkpoints` folder.