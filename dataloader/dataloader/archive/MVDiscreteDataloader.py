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

class MVDuscreteDataloader(Dataset):
    def __init__(self,file_paths=None,train=True,skip=1,gtbone=False,avetemp=False,bonecoeff=1.6):
        print("loading mv")
        self.train = train
        self.gtbone=gtbone
        self.demo = MultiviewDatasetDemo(loadMode=False,file_path=file_paths)
        self.num_samples = self.demo.N * 4
        self.skip=skip
        self.datasetname='MV'
        N=self.demo.joints.shape[0]
        self.bonecoeff=bonecoeff
        self.palmcoeff=2
        joints = self.demo.joints4view.copy().reshape(4 * N, 21, 4, 1)[..., :3, 0]
        joints = joints[:, MV2mano_skeidx]
        print(self.datasetname + ' scale ', joints[0,0],'gtbone',gtbone,'bonecoeff',bonecoeff)

        if avetemp:self.ref=get32fTensor(getRefJointsFromDataset(joints/1000,0))
        else:self.ref = get32fTensor(joints[0]/1000)

        self.boneLenMean, self.boneLenStd,self.curvatureMean, self.curvatureStd=\
        getBonePalmMeanStd(joints,bonecoeff=self.bonecoeff,palmcoeff=self.palmcoeff,debug=True)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if(idx%self.skip!=0):return {}
        vi = idx % 4
        id = idx // 4
        k = self.demo.Ks[vi]
        k_ = np.linalg.inv(k)
        #depth = 255 - self.demo.getBetterDepth(id, uselist=True)[vi]
        image=self.demo.getImgsList(id)[vi]
        kp_coord_xyz = self.demo.getPose3D(id, vi)/1000
        kp_coord_uvd = perspectiveProjection(kp_coord_xyz.copy(), k).astype(np.float32)
        image = image.reshape(480, 640, 3)

        kp_coord_xyz = kp_coord_xyz[MV2mano_skeidx]
        kp_coord_uvd = kp_coord_uvd[MV2mano_skeidx]

        if (self.gtbone):
            boneLenMean, boneLenStd, curvatureMean, curvatureStd = \
                getBonePalmMeanStd(kp_coord_xyz*1000,bonecoeff=self.bonecoeff,
                                   palmcoeff=self.palmcoeff,debug=True)
        else:
            boneLenMean, boneLenStd, curvatureMean, curvatureStd = \
                self.boneLenMean, self.boneLenStd, self.curvatureMean, self.curvatureStd

        # img=self.demo.drawPose4view(idx//4,view=1)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

        uv_vis = np.ones(21).astype(np.bool)
        # for i in range(kp_coord_uvd.shape[0]):
        #     image=cv2.circle(image, (kp_coord_uvd[i, 0], kp_coord_uvd[i, 1]), 3, (255,0,0))
        #     # image=cv2.putText(image,str(i),(kp_coord_uvd[i, 0], kp_coord_uvd[i, 1]),cv2.FONT_HERSHEY_SIMPLEX,
        #     #                   1,(255))
        # cv2.imshow('img', image)
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
        image_crop = cv2.resize(image_crop, (256, 256), interpolation=cv2.INTER_NEAREST)

        scaleu = 256 / (u2 - u1)
        scalev = 256 / (v2 - v1)

        dl=10
        scale = (np.array([scaleu, scalev, 1 / dl]) * np.array([64 / 256, 64 / 256, 64])).astype(np.float32)

        image_crop = image_crop.reshape([256, 256, 3])
        pose3d = kp_coord_uvd.reshape([21, 3]).copy()
        transition = np.array([-u1, -v1, dl//2]).astype(np.float32)

        # print(relative_depth.shape)
        pose3d[:, -1] = relative_depth[:, -1].copy()
        pose3d += transition
        pose3d = pose3d * scale



        if (self.train):
            image_crop, pose3d,randTrans,randScale,randAngle = \
                augment.processing_augmentation_Heatmap(image_crop,pose3d)
            pose3d = np.reshape(pose3d, [21, 3])



            # cjitter = torchvision.transforms.ColorJitter(brightness=0.8, contrast=[0.4, 1.6], saturation=[0.4, 1.6],
            #                                              hue=0.1)
            image_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        else:
            randTrans, randScale, randAngle=np.zeros([1,2]),np.ones([1,1]),np.zeros(1)
            pose3d = np.reshape(pose3d, [21,3])
            image_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        image_crop = image_trans(torchvision.transforms.ToPILImage()((image_crop).astype(np.uint8)))

        # img=(image_crop.permute(1,2,0).numpy() * 255).astype(np.uint8).copy()
        # for i in range(21):
        #     uv = ((pose3d[i, :2])*np.array([256/64,256/64])).astype(int)
        #     img = cv2.circle(img, tuple(uv), 1, (255, 255, 0))
        # cv2.imshow('o1', img)
        # cv2.waitKey(0)


        dic = {"image": image_crop, "pose_gt": get32fTensor(pose3d), 'scale': get32fTensor(scale),
               'transition': get32fTensor(transition),
               "pose3d_root": get32fTensor(pose3d_root), "scaled": get32fTensor(scaled),
               'K_': get32fTensor(k_).reshape(1, 3, 3),'K': get32fTensor(k).reshape(1, 3, 3),
               '3d': torch.tensor([1 if self.datasetname == 'RHD' else 0]).long(), 'randTrans': get32fTensor(randTrans),
               'randScale': get32fTensor(randScale), 'randAngle': get32fTensor(randAngle),
               'kp_coord_uvd': get32fTensor(kp_coord_uvd), 'kp_coord_xyz': get32fTensor(kp_coord_xyz),
               'ref':self.ref.clone(),
               'boneLenMean':boneLenMean,'boneLenStd':boneLenStd,
               'curvatureMean':curvatureMean,'curvatureStd':curvatureStd,}
        if (self.train == False and self.datasetname=='MV'): dic['imgori'] = get32fTensor(image)
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