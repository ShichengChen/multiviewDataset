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
import imageio
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
set = 'evaluation'

def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 2**8 + bottom_bits).astype('float32')
    depth_map /= float(2**16 - 1)
    depth_map *= 5.0
    return depth_map
class RHDDuscreteDataloader(Dataset):
    def __init__(self, train=True,path_name=path_to_db,gtbone=False,r50=False):
        print("loading rhd")
        if(train):self.mode='training'
        else:self.mode='evaluation'
        self.train=train
        self.num_samples=0
        self.path_name=path_name
        self.datasetname = 'RHD'
        self.gtbone=gtbone
        with open(os.path.join(self.path_name, self.mode, 'anno_%s.pickle' % self.mode), 'rb') as fi:
            self.anno_all = pickle.load(fi)
            self.num_samples = len(self.anno_all.items())
        print('RHDDateset3D self.num_samples',self.num_samples)
        self.imagesz = 64 if r50 else 256
        # cur=np.zeros([20])
        # allt=3000
        #for i in range(allt):
        # kp_coord_xyz = self.anno_all[4]['xyz'].astype(np.float32)[-21:, :].copy()
        # # RHD2mano_skeidx=[0,8,7,6, 12,11,10, 20,19,18, 16,15,14, 4,3,2,1, 5,9,13,17]
        # kp_coord_xyz=kp_coord_xyz[RHD2mano_skeidx]
        # if(self.out_skeidx is not None):
        #     kp_coord_xyz = kp_coord_xyz[self.out_skeidx]
        # self.ref=get32fTensor(kp_coord_xyz)
        #    cur += getBoneLen(kp_coord_xyz) * 1000
        #print(cur/allt)
        ''' 
        0 89.031 1 29.982 2 20.829 3 25.649 4 92.399 5 28.536 6 21.490  7 26.097 8 79.562 9 19.171 10 16.992 11 20.351 12 84.266 13 25.942 14 22.403 15 25.257 16 38.418  17 30.045 18 26.056 19 34.183 
        '''
        self.bonecoeff = 2.4
        self.palmcoeff = 1.8

        joints = []
        for i in range(len(self.anno_all)):
            if (i == 20500 or i == 28140):kp_coord_xyz = self.anno_all[0]['xyz'].astype(np.float32).copy()
            else:kp_coord_xyz = self.anno_all[i]['xyz'].astype(np.float32).copy()
            joints.append(kp_coord_xyz[-21:, :].reshape(21, 3)[RHD2mano_skeidx].reshape(1, 21, 3))
        joints = get32fTensor(np.concatenate(joints, axis=0) * 1000)
        print(self.datasetname + ' scale ', joints[0,0])

        self.boneLenMean, self.boneLenStd, self.curvatureMean, self.curvatureStd = \
            getBonePalmMeanStd(joints,bonecoeff=self.bonecoeff,palmcoeff=self.palmcoeff,debug=True)

        # if(avetemp):self.ref = get32fTensor(getRefJointsFromDataset(joints.numpy()/1000, 4))
        # else:
        self.ref = get32fTensor(joints.numpy()[4]/1000)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if (idx == 20500 or idx == 28140): idx = 0
        image = imageio.imread(os.path.join(path_to_db, self.mode, 'color', '%.5d.png' % idx))
        mask = imageio.imread(os.path.join(path_to_db, self.mode, 'mask', '%.5d.png' % idx))
        #depth = scipy.misc.imread(os.path.join(path_to_db, self.mode, 'depth', '%.5d.png' % idx))
        #print('depth.shape', depth.shape)
        #depth = depth_two_uint8_to_float(depth[:, :, 0], depth[:, :, 1])
        anno=self.anno_all[idx]

        # kp_coord_uv = anno['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel
        kp_visible = anno['uv_vis'][:, 2] == 1  # visibility of the keypoints, boolean
        kp_coord_xyz = anno['xyz'].astype(np.float32).copy()  # x, y, z coordinates of the keypoints, in meters
        K=k = anno['K'].astype(np.float32)  # matrix containing intrinsic parameters

        kp_coord_uvd= perspectiveProjection(kp_coord_xyz,K)
        #print(kp_coord_xyz[0],kp_coord_uvd[0],K)
        k_ = np.linalg.inv(k)


        cond_l = np.logical_and(mask > 1, mask < 18)
        cond_r = mask > 17
        num_px_left_hand = np.sum(cond_l)
        num_px_right_hand = np.sum(cond_r)
        hand_side_left = num_px_left_hand > num_px_right_hand
        if hand_side_left:
            kp_coord_xyz = kp_coord_xyz[:21, :]
        else:
            kp_coord_xyz = kp_coord_xyz[-21:, :]

        if hand_side_left:
            kp_coord_uvd = kp_coord_uvd[:21, :]
            uv_vis = kp_visible[:21]
        else:
            kp_coord_uvd = kp_coord_uvd[-21:, :]
            uv_vis = kp_visible[-21:]
        # RHD2mano_skeidx=[0,8,7,6, 12,11,10, 20,19,18, 16,15,14, 4,3,2,1, 5,9,13,17]
        kp_coord_uvd=kp_coord_uvd[RHD2mano_skeidx].copy()
        uv_vis=uv_vis[RHD2mano_skeidx].copy()
        kp_coord_xyz=kp_coord_xyz[RHD2mano_skeidx].copy()


        if(hand_side_left):
            image = cv2.flip(image, 1)
            kp_coord_xyz[:, 0] = -kp_coord_xyz[:, 0]
            kp_coord_uvd[:,0]=image.shape[1]-kp_coord_uvd[:,0]


        if (self.gtbone):
            boneLenMean, boneLenStd, curvatureMean, curvatureStd = \
                getBonePalmMeanStd(kp_coord_xyz*1000,bonecoeff=self.bonecoeff,
                                   palmcoeff=self.palmcoeff,debug=True)
        else:
            boneLenMean, boneLenStd, curvatureMean, curvatureStd = \
                self.boneLenMean, self.boneLenStd, self.curvatureMean, self.curvatureStd
        # kp_coord_uvd = perspectiveProjection(kp_coord_xyz, K)
        # for i in range(kp_coord_uvd.shape[0]):
        #     image = cv2.circle(image, (int(kp_coord_uvd[i, 0]), int(kp_coord_uvd[i, 1])), 2, (255, 255, 0))
        #     image = cv2.putText(image, str(i), (int(kp_coord_uvd[i, 0]), int(kp_coord_uvd[i, 1])), cv2.FONT_HERSHEY_SIMPLEX ,
        #                         1, (255,255,0), 1, cv2.LINE_AA)
        # print(idx,hand_side_left)
        # cv2.imshow("rgb hand", image)
        # cv2.waitKey(0)

        assert np.sum(np.abs(kp_coord_uvd[:,-1]-kp_coord_xyz[:,-1]))<1e-4

        pose3d = kp_coord_uvd[:21, :]
        pose3d_root = pose3d[4:5, -1:]
        pose3d_rel = pose3d - pose3d_root  # relative coords in metric coords
        index_root_bone_length = np.sqrt(np.sum((kp_coord_xyz[4, :] - kp_coord_xyz[5, :])**2))
        scaled = index_root_bone_length
        relative_depth = (pose3d_rel / scaled)
        #print(np.max(np.abs(relative_depth[:,-1])))


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

        dl=10
        scale = (np.array([scaleu, scalev, 1 / dl]) * np.array([64 / self.imagesz, 64 / self.imagesz, 64])).astype(np.float32)

        image_crop = image_crop.reshape([self.imagesz, self.imagesz, 3])
        pose3d = kp_coord_uvd.reshape([21, 3]).copy()
        transition = np.array([-u1, -v1, dl//2]).astype(np.float32)

        # print(relative_depth.shape)
        pose3d[:, -1] = relative_depth[:, -1].copy()
        pose3d += transition
        pose3d = pose3d * scale

        if (self.train):
            image_crop, pose3d,randTrans,randScale,randAngle = \
                augment.processing_augmentation_Heatmap(image_crop,pose3d,ImageSize=self.imagesz)
            pose3d = np.reshape(pose3d, [21, 3])

            # img = (image_crop).astype(np.uint8).copy()
            # for i in range(21):
            #     uv = ((pose3d[i, :2])*np.array([self.imagesz/64,self.imagesz/64])).astype(int)
            #     img = cv2.circle(img, tuple(uv), 1, (255, 255, 0))
            # cv2.imshow('o1', img)
            # cv2.waitKey(0)

            # cjitter = torchvision.transforms.ColorJitter(brightness=0.8, contrast=[0.4, 1.6], saturation=[0.4, 1.6],
            #                                              hue=0.1)
            image_trans = torchvision.transforms.Compose([ torchvision.transforms.ToTensor()])
        else:
            randTrans, randScale, randAngle=np.zeros([1,2]),np.ones([1,1]),np.zeros(1)
            pose3d = np.reshape(pose3d, [21,3])
            image_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        image_crop = image_trans(torchvision.transforms.ToPILImage()((image_crop).astype(np.uint8)))

        #print('image_crop', image_crop.shape)
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
               'ref':get32fTensor(kp_coord_xyz),
               'boneLenMean':boneLenMean,'boneLenStd':boneLenStd,
               'curvatureMean':curvatureMean,'curvatureStd':curvatureStd,}
        if (self.train == False and self.datasetname == 'MV'): dic['imgori'] = get32fTensor(image)
        return dic



if __name__ == '__main__':

    train_dataset=RHDDuscreteDataloader(train=True)
    for i in range(10):
        train_dataset.__getitem__(i)
