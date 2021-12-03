from __future__ import print_function, unicode_literals

import sys

import scipy.io
import scipy.misc
from cscPy.mano.network.Const import *
import cscPy.dataAugment.augment as augment
#sys.path.append('..')
import pickle
import os
from torch.utils.data import Dataset
from cscPy.mano.network.utils import *
from cscPy.Const.const import *
#from cscPy.mano.network.manolayer import VPoser,MANO_SMPL
import tqdm
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

# SET THIS to where RHD is located on your machine
path_to_db = './RHD_published_v2/'
if(not os.path.exists(path_to_db)):
    path_to_db = '/mnt/data/shicheng/RHD_published_v2/'
if(not os.path.exists(path_to_db)):
    path_to_db = '/home/csc/dataset/RHD_published_v2/'
if (not os.path.exists(path_to_db)):
    path_to_db = '/mnt/ssd/csc/RHD_published_v2/'
if (not os.path.exists(path_to_db)):
    path_to_db = '/mnt/data/csc/RHD_published_v2/'
    #os.environ["DISPLAY"] = "localhost:11.0"

set = 'training'

path_to_db = '/mnt/data/shicheng/STB/'
if(not os.path.exists(path_to_db)):
    path_to_db = '/home/csc/dataset/'
if(not os.path.exists(path_to_db)):
    path_to_db = '/mnt/ssd/csc/STB/'
if(not os.path.exists(path_to_db)):
    path_to_db = '/mnt/data/csc/STB/'

R = np.array([0.00531,-0.01196,0.00301])
T = np.array([-24.0381, -0.4563, -1.2326])

#for rgb image
fx = 607.92271
fy = 607.88192
tx = 314.78337
ty = 236.42484
K = np.array([[fx,0,tx],[0,fy,ty],[0,0,1]])
import cv2
rotationMatrix = cv2.Rodrigues(R)[0]
T = np.reshape(T,[1,1,1,3,1])

imageNum = 1500

class STBDiscretedDataloader(Dataset):
    def __init__(self,train=True,setNum=10,skip=1,gtbone=False,avetemp=False,bonecoeff=1.6):
        assert avetemp==False,"not implement"
        if(setNum==2):
            self.sequences=['B1Counting', 'B1Random']
        elif(setNum==10):
            self.sequences = ['B2Counting', 'B2Random', 'B3Counting', 'B3Random', 'B4Counting', 'B4Random',
                              'B5Counting', 'B5Random', 'B6Counting', 'B6Random']
        elif(setNum==12):
            self.sequences = ['B2Counting', 'B2Random', 'B3Counting', 'B3Random', 'B4Counting', 'B4Random',
                              'B5Counting', 'B5Random', 'B6Counting', 'B6Random','B1Counting', 'B1Random']
        else:assert False
        self.handPara=[]
        self.train=train
        self.gtbone=gtbone
        self.datasetname = 'STB'
        for seq in self.sequences:
            matfile = '%slabels/%s_SK.mat' % (path_to_db, seq)
            data = scipy.io.loadmat(matfile)
            self.handPara.append(data['handPara'])
        self.handPara=np.array(self.handPara).transpose(0,3,2,1).astype(np.float32)
        self.bonecoeff = bonecoeff
        self.palmcoeff = 2
        wrist_xyz = self.handPara[:,:,16:17, :] + 1.43 * (self.handPara[:,:,0:1, :] - self.handPara[:,:,16:17, :])
        self.handPara = np.concatenate([wrist_xyz, self.handPara[:,:,1:, :]], axis=2)
        self.handPara = self.handPara[:,:,STB2Bighand_skeidx, :][:,:,Bighand2mano_skeidx, :]



        self.handPara=np.expand_dims(self.handPara,axis=4)
        #print('self.handPara',self.handPara.shape)
        self.handPara = (np.transpose(rotationMatrix) @ (self.handPara - T))[...,0] / 1000
        self.handPara[:, :, :, 0] = -self.handPara[:, :, :, 0]  # left hand to right hand
        joints=self.handPara.copy()

        print('STB scale', self.handPara[0, 0, 0, :], len(self.sequences) * imageNum,'bonecoeff',bonecoeff)
        self.ref=get32fTensor(joints[0,0])
        # if avetemp:self.ref=get32fTensor(getRefJointsFromDataset(joints/1000,0))
        # else:self.ref = get32fTensor(joints[0]/1000)

        self.boneLenMean, self.boneLenStd,self.curvatureMean, self.curvatureStd=\
        getBonePalmMeanStd(joints.reshape(-1,21,3),bonecoeff=self.bonecoeff,palmcoeff=self.palmcoeff,debug=True)



    def __len__(self):
        return len(self.sequences)*imageNum

    def __getitem__(self, idx):
        folder_idx=idx//imageNum
        id=idx%imageNum
        img_path_rgb = '%s%s/%s_%s_%d.png' % (path_to_db, self.sequences[folder_idx], 'SK', 'color', id)
        image = scipy.misc.imread(img_path_rgb)
        kp_coord_xyz=self.handPara[folder_idx,id].copy()
        k=K
        k_ = np.linalg.inv(k)
        kp_coord_uvd = perspectiveProjection(kp_coord_xyz.copy(), k).astype(np.float32)
        image = image.reshape(480, 640, 3)
        image = cv2.flip(image, 1)

        if (self.gtbone):
            boneLenMean, boneLenStd, curvatureMean, curvatureStd = \
                getBonePalmMeanStd(kp_coord_xyz * 1000, bonecoeff=self.bonecoeff,
                                   palmcoeff=self.palmcoeff, debug=True)
        else:
            boneLenMean, boneLenStd, curvatureMean, curvatureStd = \
                self.boneLenMean, self.boneLenStd, self.curvatureMean, self.curvatureStd

        uv_vis = np.ones(21).astype(np.bool)

        # print("uvd", kp_coord_uvd)
        # for i in range(kp_coord_uvd.shape[0]):
        #     image=cv2.circle(image, (kp_coord_uvd[i, 0], kp_coord_uvd[i, 1]), 3, (255,0,0))
        #     image=cv2.putText(image,str(i),(kp_coord_uvd[i, 0], kp_coord_uvd[i, 1]),cv2.FONT_HERSHEY_SIMPLEX,
        #                       1,(255))
        # cv2.imshow('img', image)
        # cv2.waitKey(0)

        assert np.sum(np.abs(kp_coord_uvd[:, -1] - kp_coord_xyz[:, -1])) < 1e-4
        

        pose3d = kp_coord_uvd[:21, :]
        pose3d_root = pose3d[4:5, -1:]
        pose3d_rel = pose3d - pose3d_root  # relative coords in metric coords
        index_root_bone_length = np.sqrt(np.sum((kp_coord_xyz[4, :] - kp_coord_xyz[5, :]) ** 2))
        scaled = index_root_bone_length
        relative_depth = (pose3d_rel / scaled)
        # print(np.max(np.abs(relative_depth[:,-1])))

        pose_uv_all = kp_coord_uvd[:21, :2]
        crop_center = pose_uv_all[4, :2]
        crop_center = np.reshape(crop_center, 2)
        pose_uv_vis = pose_uv_all[uv_vis, :]
        crop_size = np.max(np.absolute(pose_uv_vis - crop_center) * 1.2)
        crop_size = np.minimum(np.maximum(crop_size, 25.0), 200.0)
        image_crop, (u1, v1, u2, v2) = imcrop(image, crop_center, crop_size)
        image_crop = cv2.resize(image_crop, (256, 256), interpolation=cv2.INTER_NEAREST)

        scaleu = 256 / (u2 - u1)
        scalev = 256 / (v2 - v1)

        dl = 10
        scale = (np.array([scaleu, scalev, 1 / dl]) * np.array([64 / 256, 64 / 256, 64])).astype(np.float32)

        image_crop = image_crop.reshape([256, 256, 3])
        pose3d = kp_coord_uvd.reshape([21, 3]).copy()
        transition = np.array([-u1, -v1, dl // 2]).astype(np.float32)

        # print(relative_depth.shape)
        pose3d[:, -1] = relative_depth[:, -1].copy()
        pose3d += transition
        pose3d = pose3d * scale

        if (self.train):
            image_crop, pose3d, randTrans, randScale, randAngle = \
                augment.processing_augmentation_Heatmap(image_crop, pose3d)
            pose3d = np.reshape(pose3d, [21, 3])
            # cjitter = torchvision.transforms.ColorJitter(brightness=0.8, contrast=[0.4, 1.6], saturation=[0.4, 1.6],
            #                                              hue=0.1)
            image_trans = torchvision.transforms.Compose([ torchvision.transforms.ToTensor()])
        else:
            randTrans, randScale, randAngle = np.zeros([1, 2]), np.ones([1, 1]), np.zeros(1)
            pose3d = np.reshape(pose3d, [21, 3])
            image_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        image_crop = image_trans(torchvision.transforms.ToPILImage()((image_crop).astype(np.uint8)))

        # print('image_crop', image_crop.shape)
        # img=(image_crop.permute(1,2,0).numpy() * 255).astype(np.uint8).copy()
        # cv2.imshow('o1', img)
        # cv2.waitKey(0)

        dic = {"image": image_crop, "pose_gt": get32fTensor(pose3d), 'scale': get32fTensor(scale),
               'transition': get32fTensor(transition),
               "pose3d_root": get32fTensor(pose3d_root), "scaled": get32fTensor(scaled),
               'K_': get32fTensor(k_).reshape(1, 3, 3),'K': get32fTensor(k).reshape(1, 3, 3),
               '3d': torch.tensor([1 if self.datasetname == 'RHD' else 0]).long(), 'randTrans': get32fTensor(randTrans),
               'randScale': get32fTensor(randScale), 'randAngle': get32fTensor(randAngle),
               'kp_coord_uvd': get32fTensor(kp_coord_uvd), 'kp_coord_xyz': get32fTensor(kp_coord_xyz),
               'ref': self.ref.clone(),
               'boneLenMean': boneLenMean, 'boneLenStd': boneLenStd,
               'curvatureMean': curvatureMean, 'curvatureStd': curvatureStd, }
        if (self.train == False and self.datasetname=='MV'): dic['imgori'] = get32fTensor(image)
        return dic

if __name__ == '__main__':
    STBDiscretedDateset3D.__getitem__(0)
    pass



