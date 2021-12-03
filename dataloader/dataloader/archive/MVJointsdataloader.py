from __future__ import print_function, unicode_literals
import sys
sys.path.append('..')
import numpy as np
from cscPy.globalCamera.camera import perspective_back_projection, CameraIntrinsics, CameraSeries,perspective_projection
import cscPy.dataAugment.augment as augment
import torchvision
from torch.utils.data import Dataset
import cv2
from torch.utils.data import Dataset
from cscPy.multiviewDataset.toolkit import MultiviewDatasetDemo
from cscPy.handProcess.dataAugment import processing_augmentation
from cscPy.handProcess.preprocessImages import preprocessMVdataset,imcrop
import torchvision
import cv2
from cscPy.mano.network.utils import *
from cscPy.mano.network.utilsSmallFunctions import *
from cscPy.mano.network.Const import *
from cscPy.globalCamera.camera import CameraIntrinsics,perspective_projection

import matplotlib.pyplot as plt

class MVJointsDataloader(Dataset):
    def __init__(self,file_paths=None,train=True):
        print("loading mv")
        self.train = train
        self.demo = MultiviewDatasetDemo(loadMode=False,file_path=file_paths)
        self.num_samples = self.demo.N * 4
        self.datasetname='MV'
        N=self.demo.joints.shape[0]


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        vi = idx % 4
        id = idx // 4
        kp_coord_xyz = self.demo.getPose3D(id, vi)/1000
        kp_coord_xyz = kp_coord_xyz[MV2mano_skeidx]

        pose3d = kp_coord_xyz[:21, :]
        pose3d_root = pose3d[4:5, :].copy()
        pose3d_rel = pose3d - pose3d_root  # relative coords in metric coords
        #scaled = np.sqrt(np.sum((kp_coord_xyz[4, :] - kp_coord_xyz[5, :]) ** 2))
        scaled = 1
        pose3d = (pose3d_rel / scaled)

        if (self.train):
            _, pose3d = augment.processing_augmentation_RGB(None, pose3d)
            pose3d = np.reshape(pose3d, [21, 3])
        else:
            pose3d = np.reshape(pose3d, [21, 3])

        dic = {"pose_gt": get32fTensor(pose3d.reshape(21, 3)),
               "pose3d_root": get32fTensor(pose3d_root.reshape(1, 3)),
               "scaled": get32fTensor(np.array([scaled]).reshape(1, 1)),
               }
        return dic
if __name__ == '__main__':
    mvdataloader = MVDuscreteDataloader()
    # hole=RandomHoles(640,480)
    for i in range(10):
        dms = mvdataloader.__getitem__(i)

    '''
    boneidx=[[0,1],[1,2],[2,3],[3,4],
         [0,5],[5,6],[6,7],[7,8],
         [0,9],[9,10],[10,11],[11,12],
         [0,13],[13,14],[14,15],[15,16],
         [0,17],[17,18],[18,19],[19,20],
         ]
    joints=self.demo.getPose3D(1000, 0).copy()
    for i,j in boneidx:
        print(vector_dis(joints[i],joints[j]))


    102.94658809309807
    21.794494717932782
    20.71231517744938
    11.357816692040773
    102.90772565755208
    23.43074902793336
    21.000000000238096
    17.00000000029412
    92.56889326339599
    13.892443989809713
    25.0000000002
    18.78829422832206
    90.22194854917511
    10.049875621618408
    22.583179581493834
    13.928388277543098
    36.90528417462735
    31.464265445263457
    23.727621035620068
    26.925824035858216

    Process finished with exit code 0

    '''