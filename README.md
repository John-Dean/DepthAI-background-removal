# DepthAI Background Removal
This repo shows off a way to isolate foreground objects from the scene background using a DepthAI camera.

## Requirements
Tested on a Windows 11 AMD 5950x Nvidia 3090 machine running:
* Python 3.9.9
* numpy 1.22.0
* open3d 0.14.1.0
* depthai 2.14.1.0

## How to install
1. Clone the repo
2. Open a Python terminal in the root directory of the repo
3. Run the following to install the dependencies  
   ```python3 install_requirements.py```

## How to run
Run the following:  
`python3 main.py`  
This should spawn a window of the depth camera output, where the foreground objects are coloured blue, and the background of the scene is coloured green.  

As you move around the scene, the background should fill in where you have moved if not already present. This allows for an easier time running voxel based neural networks as it removes a lot of the background for you.
