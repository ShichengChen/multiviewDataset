import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import os,pickle,imageio,argparse,sys
#from util_lmdb import util_lmdb
from torch import nn
import torchvision.models as models
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import torch
from cscPy.globalCamera.util import fetch_all_sequences,load_rgb_maps,load_depth_maps,get_cameras_from_dir
from cscPy.globalCamera.camera import CameraIntrinsics,perspective_projection,perspective_back_projection
from cscPy.globalCamera.constant import Constant
from cscPy.dataloader.proprecess.preprocessJoints import preprocessJoints
import numpy as np


from tqdm import tqdm
RHD2Bighand_skeidx = [0,4,8,12,16,20,3,2,1,7,6,5,11,10,9,15,14,13,19,18,17]
MF2RHD=[0,20,19,18,17,4,3,2,1,8,7,6,5,12,11,10,9,16,15,14,13]
key_points = [7]+[11, 14, 16, 15]+[17, 20, 22, 21]+[23, 26, 28, 27]+[29, 32, 34, 33]+[35,37,40,39]
#key_points = [7]+[12, 14, 16, 15]+[18, 20, 22, 21]+[24, 26, 28, 27]+[30, 32, 34, 33]+[35,37,40,39]
kpbalance0 = [6]
kpbalance1 = [12, 11, 13, 15]+[18, 17, 19, 21]+[24, 23, 25, 27]+[30, 29, 31, 33]
kpbalance2 = [36,38,40,39]

class MF3D(Dataset):
    def __init__(self,file_path,adjust=True,onlygt=True,usedirect=True,idxl=None,idxr=None):
        baseDir = file_path[:file_path.rfind('/', 0, file_path.rfind('/'))]
        date = baseDir[baseDir.rfind('/') + 1:]
        self.date=date
        self.baseDir=baseDir
        truemaskjoints = np.load(os.path.join(baseDir, '1_maskjoint.npy'))[100:]

        xyz_centers = np.load(os.path.join(baseDir, 'mlresults', '1_xyz_centers.npy'))
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
            joints = results['est']
        joints = np.array(joints)

        calib_path = os.path.join(baseDir, 'calib.pkl')

        cam_list = get_cameras_from_dir(baseDir[:baseDir.rfind('/')], baseDir[baseDir.rfind('/') + 1:], "1")
        cam_list.sort()
        with open(calib_path, 'rb') as f:
            camera_pose_map = pickle.load(f)
        camera, camera_pose = [], []
        for camera_ser in cam_list:
            camera.append(CameraIntrinsics[camera_ser])
            camera_pose.append(camera_pose_map[camera_ser])

        for i in range(4):
            if (np.allclose(camera_pose[i], np.eye(4))):
                rootcameraidx = i
        if (joints.shape[-1] == 3 and joints.shape[-2] == 21):
            numberofframes = joints.shape[0]
            avecanjoints = np.expand_dims(np.concatenate([joints, np.ones([joints.shape[0], 21, 1])], axis=2), axis=3)
        else:
            joints = joints.transpose((1, 0, 2, 3, 4))
            numberofframes = joints.shape[1]
            xyz_centers = np.concatenate([xyz_centers, np.zeros([xyz_centers.shape[0], xyz_centers.shape[1], 1])],
                                         axis=2)
            xyz_centers = np.expand_dims(xyz_centers, axis=-1)
            xyz_centers = xyz_centers.transpose((1, 0, 2, 3))
            joints = joints + np.expand_dims(xyz_centers, axis=2)
            c0 = joints[:, :, key_points[:1]]# * (1 - 1 * 0.1) + joints[:, :, kpbalance0] * 1 * 0.1
            c1 = joints[:, :, key_points[1:17]]# * (1 - 0.1 * 1) + joints[:, :, kpbalance1] * 1 * 0.1
            c2 = joints[:, :, key_points[17:]]# * (1 - 0.1 * 1) + joints[:, :, kpbalance2] * 1 * 0.1
            predictjoints = np.concatenate([c0, c1, c2], axis=2)
            canjoints = (-1) * np.ones((4, numberofframes, len(key_points), 4, 1)).astype(np.int64)
            for dev_idx, rs_dev in enumerate(cam_list):
                canjoints[dev_idx] = camera_pose[dev_idx] @ predictjoints[dev_idx]
            avecanjoints = np.median(canjoints, axis=0)

        with open(os.path.join(baseDir, 'mlresults', date + '-joints.npy'), 'wb') as f:
            np.save(f, avecanjoints)


        joints = np.load(os.path.join(baseDir, "mlresults", date+'-joints.npy'))

        joints=joints[...,:3,0].astype(np.float32)

        if(idxl is not None and idxr is not None):
            joints=joints[idxl:idxr]

        self.joints=joints
        n=self.joints.shape[0]

        if(onlygt):
            self.joints = self.joints[truemaskjoints[:, 0, 0, 0]]
        self.num_samples=self.joints.shape[0]
        self.usedirect=usedirect
        if (usedirect):
            self.inshape = 41
        else:
            self.inshape = 21
        self.adjust=adjust

    def __len__(self):
        return self.num_samples#//400

    def __getitem__(self, idx):
        kp_coord_xyz = self.joints[idx]
        return np.array([idx]),preprocessJoints(kp_coord_xyz,adjust=self.adjust,usedirect=self.usedirect)


