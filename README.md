# Multiview Dataset
## Provided data
- four views color images 
- four views depth images
- intrinsic and extrinsic camera parameters
- 21 hand joints
    - 0 wrist
    - 1 mcp index, 2 pip index, 3 dip index, 4 tip index
    - 5 mcp middle, 6 pip middle, 7 dip middle, 8 tip middle
    - 5 mcp ring, 6 pip ring, 7 dip ring, 8 tip ring
    - 5 mcp pinky, 6 pip pinky, 7 dip pinky, 8 tip pinky
    - 5 mcp thumb, 6 pip thumb, 7 dip thumb, 8 tip thumb
- mano parameters

## Access the dataset
- we also provide a script for reading the data
- getCameraPose: get camera extrinsic parameters
- getCameraIntrinsic: get camera intrinsic parameters
- readRGB(ith,iv): get the ith rgb image of the iv view
- readDepth(ith,iv): get the ith depth image of the iv view
- getManoParam: get mano parameter
- get4viewCloud: get point cloud generated from depth
- code is [here](https://github.com/ShichengChen/ChenPyLib)


## Info for our camera calibration
- [here](https://github.com/ShichengChen/multiviewDataset/camera-calibration)

    
