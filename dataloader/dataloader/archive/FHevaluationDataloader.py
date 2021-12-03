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
from cscPy.handProcess.preprocessImages import preprocessMVdataset,imcrop,imcrop2
import cv2
import json
import imageio
path_to_db = "/mnt/ssd/csc/freihand/"
if(not os.path.exists(path_to_db)):
    path_to_db = "/home/csc/dataset/FreiHAND_pub_v2/"
import json
class FreihandEvaluationDateset(Dataset):#'/mnt/data/csc/freihand/'
    def __init__(self, path_name=path_to_db,train=True,r50=False):#,rootidx=4):
        self.test=False
        #self.rootidx=rootidx
        self.train = train
        self.path_name = path_name
        self.imagesz=64 if r50 else 256

        print("load Freihand",'r50',r50)
        self.datasetname = 'FH'
        #32560 each set 4 set totolly
        with open(os.path.join(self.path_name, 'evaluation_K.json'), 'r') as fid:
            self.anno_k = json.load(fid)
        with open(os.path.join(self.path_name, 'evaluation_scale.json'), 'r') as fid:
            self.anno_scale = json.load(fid)
        self.pathlist = [os.path.join(self.path_name, 'evaluation/rgb', '%.8d.jpg' % i) for i in range(len(self.anno_k))]
        bpath=os.path.join(path_to_db,'bbox_root_freihand_output.json')
        #bbox=json.loads()
        self.bboxs=[]
        self.xyzroot=[]
        with open(bpath) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines][0]
            lines=json.loads(lines)
            #print(lines)
            for d in lines:
                self.bboxs.append(np.array([d['bbox']]))
                self.xyzroot.append(np.array([d['root_cam']]).reshape(1,3))

    def __len__(self):
        return len(self.pathlist)

    def __getitem__(self, idx):
        image = imageio.imread(self.pathlist[idx])
        k=np.array(self.anno_k[idx])
        bbox=self.bboxs[idx][0]
        k_ = np.linalg.inv(k)
        # x1 = int(np.round(bbox[0] - bbox[2]/2))
        # y1 = int(np.round(bbox[1] - bbox[3]/2))
        # x2 = int(np.round(bbox[0] + bbox[2]/2))
        # y2 = int(np.round(bbox[1] + bbox[3]/2))
        x1 = max(min(int(bbox[0]),int(bbox[2]))-0,0)
        y1 = max(min(int(bbox[1]),int(bbox[3]))-0,0)
        x2 = min(max(int(bbox[0]),int(bbox[2]))+20,223)
        y2 = min(max(int(bbox[1]),int(bbox[3]))+20,223)

        xyzroot = self.xyzroot[idx]
        uvdroot = perspectiveProjection(xyzroot, k).reshape(-1)
        #print(x1,y1,x2,y2,bbox)
        # image=cv2.rectangle(image, (x1,y1),(x2,y2), (255,0,0))
        # image=cv2.circle(image, (int(uvdroot[0]),int(uvdroot[1])), 5,(255,0,0))
        # cv2.imshow('img', image)
        # cv2.waitKey(0)



        image = image.reshape(224, 224, 3)
        image224 = image.copy()
        scaled=self.anno_scale[idx]
        pose3d_root=np.array([uvdroot[-1]]).reshape(1,1)

        image_crop, (u1, v1, u2, v2) = imcrop2(image, x1=x1,y1=y1,x2=x2,y2=y2)
        image_crop = cv2.resize(image_crop, (self.imagesz, self.imagesz), interpolation=cv2.INTER_NEAREST)

        scaleu = self.imagesz / (u2 - u1)
        scalev = self.imagesz / (v2 - v1)


        dl = 10
        # if (self.rootidx == 0): dl = 20
        scale = (np.array([scaleu, scalev, 1 / dl]) * np.array([64 / self.imagesz, 64 / self.imagesz, 64])).astype(np.float32)

        image_crop = image_crop.reshape([self.imagesz, self.imagesz, 3])
        transition = np.array([-u1, -v1, dl // 2]).astype(np.float32)


        # img = (image_crop).astype(np.uint8).copy()
        # cv2.imshow('o1', img)
        # cv2.waitKey(0)

        randTrans, randScale, randAngle = np.zeros([1, 2]), np.ones([1, 1]), np.zeros(1)
        image_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                                                                       (0.229, 0.224, 0.225)),
                                                      ])

        image_crop = image_trans(torchvision.transforms.ToPILImage()((image_crop).astype(np.uint8)))

        # print('image_crop', image_crop.shape)
        # img=(image_crop.permute(1,2,0).numpy() * 255).astype(np.uint8).copy()
        # cv2.imshow('o1', img)
        # cv2.waitKey(0)

        dic = {"image": image_crop, 'scale': get32fTensor(scale),
               'transition': get32fTensor(transition),
               "pose3d_root": get32fTensor(pose3d_root), "scaled": get32fTensor(scaled),
               'K_': get32fTensor(k_).reshape(1, 3, 3),'K': get32fTensor(k).reshape(1, 3, 3),
               '3d': torch.tensor([1 if self.datasetname == 'RHD' else 0]).long(), 'randTrans': get32fTensor(randTrans),
               'randScale': get32fTensor(randScale), 'randAngle': get32fTensor(randAngle),
               "image224":get32fTensor(image224),
                }
        return dic


if __name__ == '__main__':

    train_dataset=FreihandEvaluationDateset()
    #for i in range(878,881):
    for i in range(0,881):
        train_dataset.__getitem__(i)