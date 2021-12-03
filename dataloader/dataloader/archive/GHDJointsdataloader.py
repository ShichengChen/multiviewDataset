import os
import numpy as np
import cv2
import pickle
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy import ndimage
import torchvision, scipy.misc, imageio
from PIL import Image
import math
import cscPy.dataAugment.augment as augment
from cscPy.mano.network.Const import *
from cscPy.Const.const import *
import matplotlib.pyplot as plt
from cscPy.mano.network.utils import *
from cscPy.mano.network.utilsSmallFunctions import *
from cscPy.globalCamera.camera import *
from cscPy.mano.network.manolayer import MANO_SMPL
from cscPy.mano.network.net import VPoser
import tqdm
from cscPy.Const.const import *
from cscPy.handProcess.preprocessImages import preprocessMVdataset,imcrop
import cv2
import pickle
'''
###############################^M  #left hand
### GANerated Hands Dataset ###^M
###############################^M
frames without object: 143,449^M
frames with object: 188,050^M
^M
camera intrinsics:^M
[ 617.173,       0, 315.453,^M
        0, 617.173, 242.259,^M
        0,       0,       1]^M
^M
Directory structure:^M
* joints.png^M
* license.txt^M
* README.txt (this file)^M 
* data/^M
        * withObject / noObject^M
                * <partition>/ (s.t. every directory contains 1024 example frames)^M
^M
For every frame, the following data is provided:^M
- color:                        24-bit color image of the hand (cropped)^M
- joint_pos:            3D joint positions relative to the middle MCP joint. The values are normalized such that the length between middle finger MCP and wrist is 1.
                                        The positions are organized as a linear concatenation of the x,y,z-position of every joint (joint1_x, joint1_y, joint1_z, joint2_x, ...). ^M
                                        The order of the 21 joints is as follows: W, T0, T1, T2, T3, I0, I1, I2, I3, M0, M1, M2, M3, R0, R1, R2, R3, L0, L1, L2, L3.^M
                                        Please also see joints.png for a visual explanation.
- joint2D:                      2D joint positions in u,v image coordinates. The positions are organized as a linear concatenation of the u,v-position of every joint (joint1_u, joint1_v, joint2_u, …).^M
- joint_pos_global:     3D joint positions in the coordinate system of the original camera (before cropping)^M
- crop_params:          cropping parameters that were used to generate the color image (256 x 256 pixels) from the original image (640 x 480 pixels), ^M
                                        specified as top left corner of the bounding box (u,v) and a scaling factor
~                                                                                                                                                                                                           
~                                                                                                                                                                                                           
~                                                                                                                         
'''
class GHDDateset(Dataset):
    def __init__(self, path_name='/mnt/ssd/csc/GANeratedHands_Release/data/', train=True, type=0,traintestsplit=0.8):
        print("load GHD")
        self.train = train
        self.path_name = path_name
        typefolder=[['noObject'],['withObject'],['withObject','noObject']]
        self.anno_all=[]
        for f in typefolder[type]:
            base=os.path.join(path_name,f)
            print(base)
            for (dirpath, dirnames, filenames) in os.walk(base):
                prefix = list(set([i[:4] for i in filenames]))
                for pre in prefix:
                    cur={}
                    prepath=os.path.join(dirpath,pre)
                    with open(prepath+'_joint_pos_global.txt') as f:
                        content = f.readlines()[0]
                        content = [float(x.strip()) for x in content.split(',')]
                        cur['xyz']=np.array(content).reshape(21,3)
                    cur['path']=prepath+'_color_composed.png'
                    self.anno_all.append(cur)
        self.num_samples=len(self.anno_all)
        print("scale",self.anno_all[0]['xyz'][0],self.num_samples)
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        anno = self.anno_all[idx]
        pose3d = anno['xyz']


        ghd2rhd = np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17])
        pose3d=pose3d[ghd2rhd][RHD2mano_skeidx]/1000
        pose3d[:, 0] = -pose3d[:, 0]#left hand to right hand

        pose3d_root = pose3d[4:5, :].copy()
        pose3d_rel = pose3d - pose3d_root  # relative coords in metric coords
        #scaled = np.sqrt(np.sum((pose3d[4, :] - pose3d[5, :]) ** 2))
        scaled=1
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

    #_, hand_image, pose2d, relative_depth, uv_vis, _, _, _, _, _ = train_dataset.__getitem__(7)
    # print(GHDDateset(type=0).__len__())
    # print(GHDDateset(type=1).__len__())
    # print(GHDDateset(type=2).__len__())
    train_dataset = GHDDateset()
    _, hand_image, pose2d, relative_depth, uv_vis, _, _, _, _, _, _, _ = train_dataset.__getitem__(5)
    hand_image = hand_image.permute(1,2,0).numpy()
    hand_image = (hand_image + 1)/2
    pose2d = pose2d.numpy()

    print(hand_image.shape)
    print(pose2d.shape)


    #
    hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
    print(hand_image.shape)

    print(uv_vis.shape)
    print(uv_vis)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(hand_image)
    plt.scatter(pose2d[:, 0], pose2d[:, 1], alpha=0.6)

    plt.show()