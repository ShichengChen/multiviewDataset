import pickle

import torch.nn as nn
import io
from PIL import Image
import torchvision
from cscPy.mano.network.utils import *
from cscPy.mano.network.utilsSmallFunctions import *
from cscPy.mano.network.Const import *
from cscPy.mano.network.manoArmLayer import MANO_SMPL
import pickle
import os
from torch.utils.data import Dataset

import cv2
from cscPy.globalCamera.camera import CameraIntrinsics,perspective_projection
from cscPy.handProcess.dataAugment import processing_augmentation
from cscPy.handProcess.preprocessImages import preprocessMVdataset
from cscPy.globalCamera.util import fetch_all_sequences,load_rgb_maps,load_depth_maps,get_cameras_from_dir,visualize_better_qulity_depth_map

class ManoSynthesizer(Dataset):
    def __init__(self, train=True):
        manoPath='/home/csc/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl'
        if not os.path.exists(manoPath):
            manoPath = '/home/shicheng/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl'
        self.mano_right = MANO_SMPL(manoPath, ncomps=45, oriorder=True,
                               device='cpu',userotJoints=True)
        self.faces=self.mano_right.faces
        self.ks,self.cameras=[],[]
        self.train=train
        for i in ['840412062035','840412062037','840412062038','840412062076']:
            k=np.eye(3)
            k[0,0]=CameraIntrinsics[i].fx
            k[1,1]=CameraIntrinsics[i].fy
            k[0,2]=CameraIntrinsics[i].cx
            k[1,2]=CameraIntrinsics[i].cy
            self.cameras.append(CameraIntrinsics[i])
            self.ks.append(k.copy())
        # boneidx=[[0,1],[1,2],[2,3],[3,17],
        #          [0,4],[4,5],[5,6],[6,18],
        #          [0,10],[10,11],[11,12],[12,19],
        #          [0,7],[7,8],[8,9],[9,20],
        #          [0,13],[13,14],[14,15],[15,16]]
        # for i,j in boneidx:
        #     print(i,j,vector_dis(self.mano_right.J[i],self.mano_right.J[j])*1000)
        '''
        0 1 90.64222517119393
    1 2 32.99850850565244
    2 3 22.194216379270816
    3 17 25.175832099793247
    0 4 94.73134124771708
    4 5 31.735191110706406
    5 6 23.27555890998097
    6 18 26.46532274773419
    0 10 86.06878617343956
    10 11 28.84375765686435
    11 12 24.78588129309281
    12 19 25.698042053429543
    0 7 81.83852115822049
    7 8 21.111893642852117
    8 9 18.946876040831896
    9 20 20.331808586894116
    0 13 38.565440403874135
    13 14 30.790311758302956
    14 15 27.094148631609016
    15 16 34.69535759009966
        '''

    def __len__(self):
        return 5300*4

    def __getitem__(self, idx):
        #torch.manual_seed(idx%1000)
        pose = torch.tensor(np.random.uniform(-1, 1, [1, 45]).astype(np.float32))
        pose = torch.rand([1,45],dtype=torch.float32)*2-1
        #pose = torch.tensor(np.random.uniform(-0, 0, [1, 45]).astype(np.float32))
        #rootr = torch.tensor(np.random.uniform(-3.14, 3.14, [3]).astype(np.float32))
        rootr = torch.rand([3],dtype=torch.float32)*3.14*2-3.14
        #rootr = torch.tensor(np.random.uniform(-0, 0, [3]).astype(np.float32))
        # self.avemmcp [104.49264151,-39.60188679,594.01188679]
        # median [[102,-42,597]]
        # std [[31.03781234] [30.23725049] [28.52284834]]
        vertex_gt, joint_gt = self.mano_right.get_mano_vertices(rootr.view(1, 1, 3),
                                                           pose.view(1, 45),
                                                           torch.zeros([10],dtype=torch.float32).view(1, 10),
                                                           torch.tensor([1],dtype=torch.float32).view(1, 1),
                                                           torch.tensor([[0, 0, 0]],dtype=torch.float32).view(1, 3),
                                                           pose_type='pca', mmcp_center=False)
        joint_gt=joint_gt[0].numpy()*1000
        vertex_gt=vertex_gt[0].numpy()*1000
        mmcp=joint_gt[4:5]
        #print(joint_gt)
        bir=np.random.uniform(-1,1,[3])*20
        avemmcp=np.array([[132.,76.,645.],[ 61., -70., 547.],[ -8.,  95., 608.],[102., -42., 597.]])
        vertex_gt=vertex_gt-mmcp+avemmcp[idx%4]+bir
        joint_gt=joint_gt-mmcp+avemmcp[idx%4]+bir

        import trimesh
        v = trimesh.Trimesh(vertices=vertex_gt,faces=self.faces)

        (cloud,_)=trimesh.sample.sample_surface(v,4000)
        #cloud=vertex_gt.copy()
        #print(vertex_gt.shape)

        # scene = trimesh.Scene(v)
        # scene.camera.K = self.ks[idx%4]
        # #scene.camera_transform=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        # scene.show()
        # print(scene.camera_transform)
        # data = scene.save_image(resolution=(640, 480))
        # image = np.array(Image.open(io.BytesIO(data)))
        image=np.ones([480,640])*2000
        vertex_uvd=perspective_projection(cloud,self.cameras[idx%4]).astype(int)
        pose_uv=perspective_projection(joint_gt,self.cameras[idx%4])[:,:2].astype(int)
        for i in range(vertex_uvd.shape[0]):
            c=3
            u0=np.clip(vertex_uvd[i,0]-c,0,640)
            u1=np.clip(vertex_uvd[i,0]+c,0,640)
            v0=np.clip(vertex_uvd[i,1]-c,0,480)
            v1=np.clip(vertex_uvd[i,1]+c,0,480)
            # print(vertex_uvd[i,2])
            image[v0:v1,u0:u1]=np.minimum(image[v0:v1,u0:u1],vertex_uvd[i,2])
            #image[v0:v1,u0:u1]=np.minimum(image[v0:v1,u0:u1],[vertex_uvd[i,2]])
        #vertex_uvd[:, 2] = (255 - np.clip(vertex_uvd[:, 2], 0, 2000) / 2000 * 255).astype(int)
        image=255-visualize_better_qulity_depth_map(image)
        # cv2.imshow('image', image)
        # cv2.waitKey(0  )

        # for i in range(pose_uv.shape[0]):
        #     if(i==4):
        #         cv2.circle(image,(pose_uv[i,0],pose_uv[i,1]),5,(255))
        #     cv2.circle(image, (pose_uv[i, 0], pose_uv[i, 1]), 3, (255))
        image=image.reshape(480,640,3)
        #joint_gt and cloud from mm to m
        joint_gt=joint_gt/1000
        cloud=cloud/1000
        randInidices = torch.randperm(cloud.shape[0]).numpy()
        cloud = cloud[randInidices[:906], :]
        #cloud = cloud[:906, :]
        image_crop, pose3d_normed,cloud_norm, scale, root = preprocessMVdataset(image, pose_uv, joint_gt,cloud, cidx=4, cnxt=5)


        # cv2.imshow('img', image)
        # image_crop, depth, cloud, pose3d = \
        #     processing_augmentation(image_crop, None,cloud,joint_gt,self.train)
        image_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        image_crop = image_trans(torchvision.transforms.ToPILImage()((image_crop).astype(np.uint8)))

        # cur=image_crop.permute(1,2,0).numpy()
        # cur=(cur*255).astype(np.uint8)
        # cv2.imshow('cur',cur)
        # cv2.waitKey(0)
        out = {"img": get32fTensor(image_crop.reshape(3, 256, 256)),
               "pose3d": get32fTensor(pose3d_normed).reshape(21, 3),
               "scale": get32fTensor(scale).reshape(1, 1), "root": get32fTensor(root).reshape(1, 3),
               "cloud": get32fTensor(cloud_norm).reshape(906, 3),
               "mask": torch.tensor([1]).long(),'idx':torch.tensor([idx]).long(),
               'K':get32fTensor(self.ks[idx%4]).reshape(1,3,3),'pose2d':get32fTensor(pose_uv)}
        return out




if __name__ == "__main__":
    synthesizer=ManoSynthesizer()
    #hole=RandomHoles(640,480)
    for i in range(20):
        dms=synthesizer.__getitem__(i)
        # print(dms.shape)
        # ho=hole(dms)
        # cv2.imshow('hole',ho[0].numpy())
        # cv2.waitKey(0)









# class RandomHoles(nn.Module):
#     def __init__(self, width, height):
#         super(RandomHoles, self).__init__()
#         self.sigma_sqr = 9.0  # 25.0
#         self.sigma_a = 2.0
#         self.width = width
#         self.height = height
#
#         u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
#         u_grid = torch.from_numpy(u_grid).float()
#         v_grid = torch.from_numpy(v_grid).float()
#         u_grid = u_grid.unsqueeze(dim=0)
#         v_grid = v_grid.unsqueeze(dim=0)
#         self.register_buffer('u_grid', u_grid)
#         self.register_buffer('v_grid', v_grid)
#
#     def forward(self, dms):
#         batch_num = dms.shape[0]
#         rnd = torch.ones(batch_num, 1, 1)
#         if dms.is_cuda:
#             rnd = rnd.cuda()
#         rnd_r = torch.rand_like(rnd) * self.sigma_sqr
#         rnd_a = torch.rand_like(rnd) * self.sigma_a + 1.0
#         rnd_b = torch.rand_like(rnd) * self.sigma_a + 1.0
#         rnd_u = torch.rand_like(rnd) * self.width * 0.6 + self.width * 0.2
#         rnd_v = torch.rand_like(rnd) * self.height * 0.6 + self.height * 0.2
#
#         u_grid = self.u_grid.repeat(batch_num, 1, 1)
#         v_grid = self.v_grid.repeat(batch_num, 1, 1)
#
#         dist = rnd_a * (u_grid - rnd_u) ** 2 + rnd_b * (v_grid - rnd_v) ** 2
#         mask = dist < rnd_r
#         dms[mask] = 255
#         return dms

