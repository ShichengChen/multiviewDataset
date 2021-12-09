# Multiview Dataset Toolkit
- Using multi-view cameras is a natural way to obtain a complete point cloud. However,
there is to date only one multi-view 3D hand pose dataset– NYU. Furthermore, NYU is
primarily used as a depth map dataset; although they also provided the RGB images, these
RGB images are of low resolution and quality. FreiHand also records data using a multi-
view setup, but the released images are not from corresponding viewpoints. In that sense,
it can be regarded only as a single-view dataset containing multiple views rather than a true
multi-view dataset.
- To fill this gap, we present a new multi-view RGB-D 3D hand pose dataset. We use four
RealSense D415 cameras in different views to record 4 RGB-D sequences from 4 subjects
and the resolution of our recorded dataset is 640 × 480. We use
a 21-joint model to annotate the hand pose. Additionally, we provide hand masks, 2D and
3D joint locations, hand meshes in the form of MANO parameters, real complete hand point
clouds and full camera parameters. In particular, we provide extrinsic camera parameters so
it is easy for users to use multi-view information.


## Basic setup
- download [data](https://www.dropbox.com/sh/zp2ruks8w8gegm8/AAAHEaFT70bHKJBh33e5DjfSa?dl=0)
- install basic requirements
```
pip install numpy matplotlib scikit-image transforms3d tqdm opencv-python trimesh pyrender
```
- example code
```
python toolkit.py
```


## Provided data
- four views color images 
- four views depth images
- intrinsic and extrinsic camera parameters
- 21 hand joints
    - 0 wrist
    - 1 mcp index, 2 pip index, 3 dip index, 4 tip index
    - 5 mcp middle, 6 pip middle, 7 dip middle, 8 tip middle
    - 9 mcp ring, 10 pip ring, 11 dip ring, 12 tip ring
    - 13 mcp pinky, 14 pip pinky, 15 dip pinky, 16 tip pinky
    - 17 mcp thumb, 18 pip thumb, 19 dip thumb, 20 tip thumb
- mano parameters   
![image](data/multiviewdataset.gif)
     
## Access the dataset
- data usage in [toolkit.py](https://github.com/ShichengChen/multiviewDataset/blob/main/toolkits/toolkit.py)
    - drawMesh
    - drawPose4view
    - getBetterDepth
 
## Info for our camera calibration
- [here](https://github.com/ShichengChen/multiviewDataset/tree/main/camera-calibration)

    
# Terms of use
```
@InProceedings{Local2021,
  author    = {Ziwei Yu, Linlin Yang, Shicheng Chen, Angela Yao},
  title     = {Local and Global Point Cloud Reconstruction for 3D Hand Pose Estimation},
  booktitle    = {British Machine Vision Conference (BMVC)},
  year      = {2021},
  url          = {"https://github.com/ShichengChen/multiviewDataset"}
}
```