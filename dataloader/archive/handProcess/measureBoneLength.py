import pickle

import torch.nn as nn

from cscPy.mano.network.utils import *
from cscPy.mano.network.utilsSmallFunctions import *
from cscPy.mano.network.Const import boneSpace
from cscPy.Const.const import epsilon
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

for i in range(4):
    demo = MultiviewDatasetDemo(loadMode=False,file_path=mvdatasetpaths[i])
    N=demo.joints.shape[0]
    #joints=demo.joints.copy().reshape(N,21,4,1)[...,:3,0]
    joints=demo.joints4view.copy().reshape(4*N,21,4,1)[...,:3,0]
    joints=joints[:,MV2mano_skeidx]
    out=getBoneLen(get32fTensor(joints))
    out2=getCurvature(get32fTensor(joints))
    mean, std = torch.mean(out, dim=0, keepdim=True), torch.std(out, dim=0, keepdim=True)*1.4
    mean2, std2 = torch.mean(out2, dim=0, keepdim=True), torch.std(out2, dim=0, keepdim=True)*1.8
    loss = torch.mean(torch.max(torch.abs(out - mean) - std,torch.zeros_like(out)))
    loss1 = torch.mean(torch.max(torch.abs(out2 - mean2) - std2,torch.zeros_like(out2)))
    print('idx',i)
    print(mean, std)
    print(mean2, std2)
    print('loss',loss, loss1)


path_to_db = './RHD_published_v2/'
if(not os.path.exists(path_to_db)):
    path_to_db = '/mnt/data/shicheng/RHD_published_v2/'
if(not os.path.exists(path_to_db)):
    path_to_db = '/home/csc/dataset/RHD_published_v2/'
if (not os.path.exists(path_to_db)):
    path_to_db = '/mnt/ssd/csc/RHD_published_v2/'
if (not os.path.exists(path_to_db)):
    path_to_db = '/mnt/data/csc/RHD_published_v2/'
with open(os.path.join(path_to_db, 'training', 'anno_training.pickle'), 'rb') as fi:
    anno = pickle.load(fi)

joints=[]
for i in range(len(anno)):
    if (i == 20500 or i == 28140):kp_coord_xyz = anno[0]['xyz'].astype(np.float32).copy()
    else:kp_coord_xyz = anno[i]['xyz'].astype(np.float32).copy()
    joints.append(kp_coord_xyz[-21:, :].reshape(21,3)[RHD2mano_skeidx].reshape(1,21,3))
joints=get32fTensor(np.concatenate(joints,axis=0)*1000)
out=getBoneLen(joints)
out2=getCurvature(joints)
mean, std = torch.mean(out, dim=0, keepdim=True), torch.std(out, dim=0, keepdim=True)*2.5
mean2, std2 = torch.mean(out2, dim=0, keepdim=True), torch.std(out2, dim=0, keepdim=True)*1.8
loss = torch.mean(torch.max(torch.abs(out - mean) - std,torch.zeros_like(out)))
loss1 = torch.mean(torch.max(torch.abs(out2 - mean2) - std2,torch.zeros_like(out2)))
print(mean, std)
print(mean2, std2)
print('loss',loss, loss1)

