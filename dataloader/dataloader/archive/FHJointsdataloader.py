from __future__ import print_function, unicode_literals

import sys

import scipy.io
import scipy.misc
import cscPy.dataAugment.augment as augment
import torchvision
#sys.path.append('..')
import pickle
import os
from torch.utils.data import Dataset
from cscPy.mano.network.utils import *
from cscPy.mano.network.utilsSmallFunctions import *
from cscPy.globalCamera.camera import *
from cscPy.mano.network.manolayer import MANO_SMPL
from cscPy.mano.network.net import VPoser
import tqdm
from cscPy.Const.const import *
from cscPy.handProcess.preprocessImages import preprocessMVdataset,imcrop
import cv2
import json

class FreiJointsDateset3D(Dataset):#'/mnt/data/csc/freihand/'
    def __init__(self, path_name="/mnt/ssd/csc/freihand/",train=True):
        self.test=test=False
        self.train = train
        self.path_name = path_name
        print("load Freihand")
        self.mode = "training"
        self.datasetname = 'FH'
        with open(os.path.join(self.path_name, 'training_xyz.json'), 'r') as fid:
            self.anno_xyz = json.load(fid)
        alllen=len(self.anno_xyz)
        self.pathlist,self.maskpathlist=[],[]
        for i in range(alllen):
            self.pathlist.append(os.path.join(self.path_name, 'training/rgb', '%.8d.jpg' % i))
        print('alllen',alllen)

        print('Fh', np.array(self.anno_xyz[0][0]))

    def __len__(self):
        return len(self.pathlist)

    def __getitem__(self, idx):
        #image = scipy.misc.imread(self.pathlist[idx])
        kp_coord_xyz=np.array(self.anno_xyz[idx])

        uv_vis=np.ones(21,dtype=np.bool)
        Frei2mano_skeidx=[0, 5,6,7, 9,10,11,  17,18,19, 13,14,15, 1,2,3,  4,8,12,16,20]
        #image = image.reshape(224, 224, 3)
        kp_coord_xyz = kp_coord_xyz[Frei2mano_skeidx]

        pose3d = kp_coord_xyz[:21, :]
        pose3d_root = pose3d[4:5, :].copy()
        pose3d_rel = pose3d - pose3d_root  # relative coords in metric coords
        # scaled = np.sqrt(np.sum((kp_coord_xyz[4, :] - kp_coord_xyz[5, :]) ** 2))
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

    train_dataset=FreiDateset3D(train=True)
    for i in range(100):
        train_dataset.__getitem__(i)