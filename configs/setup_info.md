This guide is for setting up the necessary environment and infrastructure to run the "Using FoundPose" commands.
 - `ssh luca-gpu`
 - (tatiana): `/opt/TurboVNC/bin/vncserver -list` --> if a VNC process is running kill it using the display id:
    - `/opt/TurboVNC/bin/vncserver -kill :1`
    - `/opt/TurboVNC/bin/vncserver` --> to run a new VNC server

(on your machine): open TurboVNC Viewer app
--> make sure to have 100.66.235.1:1 in the VNC server field. The first part is the remote host IP while the last :1 is the VNC server display id. The IP is the tailscale one for the luca-gpu machine.
--> insert the password: (see "TurboVNC Viewer" in BitWarden) => xfce desktop environment starts on the remote host.

When you are done you either kill the vnc server or exit the ssh session using `exit`.

Create the first environment `foundpose_cpu` by running:
`conda env create -f conda_foundpose_cpu.yaml`
(Here to make OPenGL running we had to install the additional dependency:
`conda install -c conda-forge libstdcxx-ng`)

This env will be used to run the first command of foundpose (generating templates)

However this command breaks the possibility to see CUDA anymore, that's why you need another conda env, equal to the foundpose_new one. Within this env you can run both the second and the third commands of foundpose.

In case tatiana passwd is asked: (see "tatiana luca-gpu" in BitWarden).

If you need to export the cond environment: `conda env export | grep -v "^prefix: " > environment.yml`