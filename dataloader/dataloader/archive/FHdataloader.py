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
import imageio
path_to_db = "/mnt/ssd/csc/freihand/"
if(not os.path.exists(path_to_db)):
    path_to_db = "/home/csc/dataset/FreiHAND_pub_v2/"

class FreiDateset3D(Dataset):#'/mnt/data/csc/freihand/'
    def __init__(self, path_name=path_to_db,train=True,setNum=4,avetemp=False,rootidx=4,gtbone=False,bonecoeff=1.6,r50=False):
        self.test=test=False
        self.train = train
        self.rootidx=rootidx
        self.r50=r50

        self.imagesz=64 if r50 else 256

        self.gtbone=gtbone
        if(self.rootidx==0):print("wrist root")
        self.path_name = path_name
        print("load Freihand")
        self.mode = "training"
        self.datasetname = 'FH'
        #32560 each set 4 set totolly
        with open(os.path.join(self.path_name, 'training_K.json'), 'r') as fid:
            self.anno_k = json.load(fid)
        with open(os.path.join(self.path_name, 'training_xyz.json'), 'r') as fid:
            self.anno_xyz = json.load(fid)

        self.pathlist = [os.path.join(self.path_name, 'training/rgb', '%.8d.jpg' % i) for i in range(len(self.anno_k))]

        # with open(os.path.join(self.path_name, 'training_scale.json'), 'r') as fid:
        #     self.anno_scale = json.load(fid)
        # curxyz = np.array(self.anno_xyz)[:100]
        # print(np.sqrt(np.sum((curxyz[:, 9]-curxyz[:,10])**2,axis=1)),self.anno_scale[:100])

        if setNum == 4:
            N = len(self.anno_k) // 5 * 4
            self.pathlist = self.pathlist[:N]
            self.anno_k = self.anno_k[:N]
            self.anno_xyz = self.anno_xyz[:N]
        elif setNum == 1:
            N = len(self.anno_k) // 5
            self.pathlist=self.pathlist[-N:]
            self.anno_k = self.anno_k[-N:]
            self.anno_xyz = self.anno_xyz[-N:]
        elif(setNum == 5):
            N = len(self.anno_k)
        else:assert False



        self.anno_xyz = np.array(self.anno_xyz)[:, Frei2mano_skeidx, :]
        #self.anno_xyz = self.anno_xyz[:, Mano2frei_skeidx, :]
        joints = self.anno_xyz.copy()

        print('sampleN', self.datasetname + ' scale ', N, joints[0, 0],'gtbone',gtbone,'bonecoeff',bonecoeff,'r50',r50)

        if avetemp:
            self.ref = get32fTensor(getRefJointsFromDataset(joints))
        else:
            self.ref = get32fTensor(joints[0])
        self.bonecoeff = bonecoeff
        self.palmcoeff = 2
        self.boneLenMean, self.boneLenStd, self.curvatureMean, self.curvatureStd = \
            getBonePalmMeanStd(joints*1000, bonecoeff=self.bonecoeff, palmcoeff=self.palmcoeff, debug=True)
    def __len__(self):
        return len(self.pathlist)

    def __getitem__(self, idx):
        image = imageio.imread(self.pathlist[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        kp_coord_xyz=np.array(self.anno_xyz[idx])
        k=np.array(self.anno_k[idx])
        kp_coord_uvd=perspectiveProjection(kp_coord_xyz,k)
        k_ = np.linalg.inv(k)

        # for i in range(kp_coord_uvd.shape[0]):
        #     image=cv2.circle(image, (int(kp_coord_uvd[i, 0]), int(kp_coord_uvd[i, 1])), 1, (255,0,0))
        #     image=cv2.putText(image,str(i),(int(kp_coord_uvd[i, 0]), int(kp_coord_uvd[i, 1])),cv2.FONT_HERSHEY_SIMPLEX,
        #                       1,(255))
        # cv2.imshow('img', image)
        # cv2.waitKey(0)
        # print(kp_coord_xyz[0],kp_coord_uvd[0],k)

        assert np.sum(np.abs(kp_coord_uvd[:, -1] - kp_coord_xyz[:, -1])) < 1e-4
        uv_vis=np.ones(21,dtype=np.bool)

        image = image.reshape(224, 224, 3)

        if (self.gtbone):
            boneLenMean, boneLenStd, curvatureMean, curvatureStd = \
                getBonePalmMeanStd(kp_coord_xyz*1000,bonecoeff=self.bonecoeff,
                                   palmcoeff=self.palmcoeff,debug=True)
        else:
            boneLenMean, boneLenStd, curvatureMean, curvatureStd = \
                self.boneLenMean, self.boneLenStd, self.curvatureMean, self.curvatureStd

        pose3d = kp_coord_uvd[:21, :]
        #pose3d_root = pose3d[self.rootidx:self.rootidx+1, -1:]
        pose3d_root = pose3d[4:5, -1:]
        pose3d_root2 = pose3d[self.rootidx:self.rootidx+1, -1:]
        pose3d_rel = pose3d - pose3d_root  # relative coords in metric coords
        index_root_bone_length = np.sqrt(np.sum((kp_coord_xyz[4, :] - kp_coord_xyz[5, :]) ** 2))
        scaled = index_root_bone_length
        #print(self.anno_scale[idx],index_root_bone_length,"anno_scale")
        relative_depth = (pose3d_rel / scaled)
        # print(np.max(np.abs(relative_depth[:,-1])))

        pose_uv_all = kp_coord_uvd[:21, :2]
        crop_center = pose_uv_all[4, :2]
        crop_center = np.reshape(crop_center, 2)
        pose_uv_vis = pose_uv_all[uv_vis, :]
        crop_size = np.max(np.absolute(pose_uv_vis - crop_center) * 1.2)
        crop_size = np.minimum(np.maximum(crop_size, 25.0), 200.0)
        image_crop, (u1, v1, u2, v2) = imcrop(image, crop_center, crop_size)
        image_crop = cv2.resize(image_crop, (self.imagesz, self.imagesz), interpolation=cv2.INTER_NEAREST)

        scaleu = self.imagesz / (u2 - u1)
        scalev = self.imagesz / (v2 - v1)

        dl = 10
        #if (self.rootidx == 0): dl = 20
        scale = (np.array([scaleu, scalev, 1 / dl]) * np.array([64 / self.imagesz, 64 / self.imagesz, 64])).astype(np.float32)

        image_crop = image_crop.reshape([self.imagesz, self.imagesz, 3])
        pose3d = kp_coord_uvd.reshape([21, 3]).copy()
        transition = np.array([-u1, -v1, dl // 2]).astype(np.float32)

        # print(relative_depth.shape)
        pose3d[:, -1] = relative_depth[:, -1].copy()
        pose3d += transition
        pose3d = pose3d * scale

        # img = (image_crop).astype(np.uint8).copy()
        # for i in range(21):
        #     uv = ((pose3d[i, :2])*np.array([self.imagesz/64,self.imagesz/64])).astype(int)
        #     img = cv2.circle(img, tuple(uv), 1, (255, 255, 0))
        # cv2.imshow('o1', img)
        # cv2.waitKey(0)



        if (self.train):
            image_crop, pose3d, randTrans, randScale, randAngle = \
                augment.processing_augmentation_Heatmap(image_crop, pose3d,ImageSize=self.imagesz)
            pose3d = np.reshape(pose3d, [21, 3])

            # img = (image_crop).astype(np.uint8).copy()
            # for i in range(21):
            #     uv = ((pose3d[i, :2])*np.array([self.imagesz/64,self.imagesz/64])).astype(int)
            #     img = cv2.circle(img, tuple(uv), 1, (255, 255, 0))
            # cv2.imshow('o1', img)
            # cv2.waitKey(0)

            cjitter = torchvision.transforms.ColorJitter(brightness=0.8, contrast=[0.4, 1.6], saturation=[0.4, 1.6],
                                                         hue=0.1)
            image_trans = torchvision.transforms.Compose([
                                                          cjitter,
                                                          torchvision.transforms.ToTensor(),
                                                          # torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                                          #                      (0.229, 0.224, 0.225)),
                                                          #torchvision.transforms.RandomErasing(),
                                                          ])
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
               "pose3d_root": get32fTensor(pose3d_root2), "scaled": get32fTensor(scaled),
               'K_': get32fTensor(k_).reshape(1, 3, 3),'K': get32fTensor(k).reshape(1, 3, 3),
               '3d': torch.tensor([1 if self.datasetname == 'RHD' else 0]).long(), 'randTrans': get32fTensor(randTrans),
               'randScale': get32fTensor(randScale), 'randAngle': get32fTensor(randAngle),
               'kp_coord_uvd': get32fTensor(kp_coord_uvd), 'kp_coord_xyz': get32fTensor(kp_coord_xyz),
               'ref': self.ref.clone(),
               'boneLenMean':boneLenMean,'boneLenStd':boneLenStd,
               'curvatureMean':curvatureMean,'curvatureStd':curvatureStd,}
        return dic


if __name__ == '__main__':

    train_dataset=FreiDateset3D(train=True)
    for i in range(100):
        train_dataset.__getitem__(i)