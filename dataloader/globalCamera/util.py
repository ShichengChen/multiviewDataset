import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
from .constant import Constant
from torch.utils.data import Dataset
import os,pickle
from dataloader.globalCamera.camera import CameraIntrinsics
import numpy as np

joint_color = [(255, 0, 0)] * 1 + \
              [(25, 255, 25)] * 4 + \
              [(212, 0, 255)] * 4 + \
              [(0, 230, 230)] * 4 + \
              [(179, 179, 0)] * 4 + \
              [(0, 0, 255)] * 4
linecolor=np.array([[25, 255, 25],[212, 0, 255],[0, 230, 230],[179, 179, 0],[0, 0, 255]])
linesg = [[0, 1, 2, 3, 17], [0, 4, 5, 6, 18], [0, 7, 8, 9, 20], [0, 10, 11, 12, 19], [0, 13, 14, 15, 16]]
RHD2Bighand_skeidx = [0,4,8,12,16,20,3,2,1,7,6,5,11,10,9,15,14,13,19,18,17]
MF2RHD=[0,20,19,18,17,4,3,2,1,8,7,6,5,12,11,10,9,16,15,14,13]
key_points = [7]+[11, 14, 16, 15]+[17, 20, 22, 21]+[23, 26, 28, 27]+[29, 32, 34, 33]+[35,37,40,39]
#key_points = [7]+[12, 14, 16, 15]+[18, 20, 22, 21]+[24, 26, 28, 27]+[30, 32, 34, 33]+[35,37,40,39]
kpbalance0 = [6]
kpbalance1 = [12, 11, 13, 15]+[18, 17, 19, 21]+[24, 23, 25, 27]+[30, 29, 31, 33]
kpbalance2 = [36,38,40,39]
epsilon=1e-6
MV2mano_skeidx=[0,1,2,3, 5,6,7, 13,14,15, 9,10,11, 17,18,19, 20,4,8,12,16]
RHD2mano_skeidx=[0,8,7,6, 12,11,10, 20,19,18, 16,15,14, 4,3,2,1, 5,9,13,17]
Frei2mano_skeidx = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]
Mano2frei_skeidx = [0, 13,14,15,16, 1,2,3,17,  4,5,6,18,  10,11,12,19, 7,8,9,20]
import os
manoPath='/home/csc/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl'
if not os.path.exists(manoPath):
    manoPath = '/home/shicheng/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl'
if not os.path.exists(manoPath):
    manoPath = '/home/csc/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl'

mvdatasetpaths=['/media/csc/Seagate Backup Plus Drive/dataset/7-14-1-2/mlresults/7-14-1-2_3_5result_45.pkl',
            '/media/csc/Seagate Backup Plus Drive/dataset/9-10-1-2/mlresults/9-10-1-2_1result_38.pkl',
            '/media/csc/Seagate Backup Plus Drive/dataset/9-17-1-2/mlresults/9-17-1-2_7result_45.pkl',
            '/media/csc/Seagate Backup Plus Drive/multicamera/9-25-1-2/mlresults/9-25-1-2_3result_45.pkl',
           ]
if not os.path.exists(mvdatasetpaths[0]):
    mvdatasetpaths = ['/mnt/data/shicheng/7-14-1-2/mlresults/7-14-1-2_3_5result_45.pkl',
                  '/mnt/data/shicheng/9-10-1-2/mlresults/9-10-1-2_1result_38.pkl',
                  '/mnt/data/shicheng/9-17-1-2/mlresults/9-17-1-2_7result_45.pkl',
                  '/mnt/data/shicheng/9-25-1-2/mlresults/9-25-1-2_3result_45.pkl',
                  ]
if not os.path.exists(mvdatasetpaths[0]):
    mvdatasetpaths = ['/mnt/ssd/shicheng/7-14-1-2/mlresults/7-14-1-2_3_5result_45.pkl',
                  '/mnt/ssd/shicheng/9-10-1-2/mlresults/9-10-1-2_1result_38.pkl',
                  '/mnt/ssd/shicheng/9-17-1-2/mlresults/9-17-1-2_7result_45.pkl',
                  '/mnt/ssd/shicheng/9-25-1-2/mlresults/9-25-1-2_3result_45.pkl',
                  ]
if not os.path.exists(mvdatasetpaths[0]):
    mvdatasetpaths = ['/mnt/ssd/csc/7-14-1-2/mlresults/7-14-1-2_3_5result_45.pkl',
                  '/mnt/ssd/csc/9-10-1-2/mlresults/9-10-1-2_1result_38.pkl',
                  '/mnt/ssd/csc/9-17-1-2/mlresults/9-17-1-2_7result_45.pkl',
                  '/mnt/ssd/csc/9-25-1-2/mlresults/9-25-1-2_3result_45.pkl',
                  ]
constant = Constant()

def visualize_better_qulity_depth_map(depth_image):
    vis_depth_image = depth_image.copy().astype(np.float32)
    vis_depth_image = np.clip(vis_depth_image,0,2000)
    vis_depth_image = vis_depth_image * 255 / 2000
    vis_depth_image[vis_depth_image < 10] = 255
    vis_depth_image=vis_depth_image*3-50
    # mask=(vis_depth_image>60)&(vis_depth_image<100)
    # vis_depth_image[mask]=vis_depth_image[mask]*3-50
    # vis_depth_image[~mask]*=2
    #print(vis_depth_image)
    vis_depth_image = np.clip(vis_depth_image, 0, 255)
    vis_depth_image = vis_depth_image.astype(np.uint8)

    return cv2.cvtColor(vis_depth_image, cv2.COLOR_GRAY2BGR)

import numpy as np
import torch

RHD2Bighand_skeidx = [0,4,8,12,16,20,3,2,1,7,6,5,11,10,9,15,14,13,19,18,17]
MF2RHD=[0,20,19,18,17,4,3,2,1,8,7,6,5,12,11,10,9,16,15,14,13]
Bighand2MF=[0,2,9,10,11,3,12,13,14,4,15,16,17,5,18,19,20,1,6,7,8]
mcp = [1, 5, 9, 13, 17]
mid = [2, 6, 10, 14, 18]
dip = [3, 7, 11, 15, 19]
tip = [4, 8, 12, 16, 20]
multiple0 = np.array([1.486806, 1.363457, 1.1637201, 1.1142688, 1.1363479]).reshape(-1, 1)
multiple1 = np.array([1.1343263, 1.1370358, 1.0368003, 1.0731235, 1.2805481]).reshape(-1, 1)
def fadjust(joints_gt,alltip=True,allmcp=True):
    if allmcp:
        d = joints_gt[mcp] - joints_gt[mid]
        joints_gt[mcp] = joints_gt[mid] + d * multiple0
    if alltip:
        d = joints_gt[tip] - joints_gt[dip]
        joints_gt[tip] = joints_gt[dip] + d * multiple1
    return joints_gt


def preprocessJoints(kp_coord_xyz,adjust=True,usedirect=True,meter2mm=1000):
    '''
    :param kp_coord_xyz: shape=21*3, length=mm
    :return: pose3d_normed, scale, pose3d_root, direct_normed
    '''
    assert (kp_coord_xyz.shape==(21,3))
    joints_gt = kp_coord_xyz[:21, :] / meter2mm
    joints_gt=joints_gt.astype(np.float32)
    if adjust: joints_gt=fadjust(joints_gt,alltip=True,allmcp=False)



    # joints_gt[:, 0] = -joints_gt[:, 0]
    direct = np.zeros([20, 3])
    ma = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
          [0, 13], [13, 14], [14, 15], [15, 16],
          [0, 17], [17, 18], [18, 19], [19, 20], ]
    for idx, i in enumerate(ma):
        direct[idx] = joints_gt[i[0]] - joints_gt[i[1]]
    direct = direct.astype(np.float32)  # *0.1

    pose3d_root = joints_gt[5, :]  # this is the root coord
    pose3d_rel = joints_gt - pose3d_root  # relative coords in metric coords
    index_root_bone_length = np.sqrt(np.sum(np.square(pose3d_rel[5, :] - pose3d_rel[1, :])))  # *3
    scale = index_root_bone_length
    pose3d_normed = pose3d_rel / scale
    pose3d_normed = pose3d_normed[MF2RHD]
    pose3d_normed = pose3d_normed[RHD2Bighand_skeidx]
    pose3d_normed = torch.from_numpy(pose3d_normed).view(21, 3)
    direct_normed = torch.from_numpy(direct / scale).view(20, 3)

    if(usedirect):
        return pose3d_normed, torch.tensor(scale), torch.tensor(pose3d_root), direct_normed
    else:
        return pose3d_normed, torch.tensor(scale), torch.tensor(pose3d_root), {}


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

        cam_list = ['840412062035','840412062037','840412062038','840412062076']
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

