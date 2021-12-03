import torch
from cscPy.dataAugment.augment import invertAugmentation
from cscPy.mano.network.utilsSmallFunctions import *
from cscPy.mano.network.utils import *
from torch.nn import functional as F

def handleheatmap(heatmaps):
    bz = heatmaps.shape[0]
    heatmaps = heatmaps.reshape((bz, 21, 64 * 64 * 64))
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((bz, 21, 64, 64, 64))
    accu_x = heatmaps.sum(dim=(2, 3)) 
    accu_y = heatmaps.sum(dim=(2, 4))
    accu_z = heatmaps.sum(dim=(3, 4))

    accu_x = accu_x * torch.arange(0, 64,device='cuda',dtype=torch.float32)
    accu_y = accu_y * torch.arange(0, 64,device='cuda',dtype=torch.float32)
    accu_z = accu_z * torch.arange(0, 64,device='cuda',dtype=torch.float32)

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)
    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
    #print(coord_out)

    return coord_out

def xyz2croppedUvd(xyz,dic):
    bs,device=xyz.shape[0],xyz.device
    scale = dic['scale'].reshape(-1, 1, 3).to(device)
    transition = dic['transition'].reshape(-1, 1, 3).to(device)
    scaled = dic['scaled'].reshape(-1, 1, 1).to(device)
    k = dic['K'].to(device)
    pose3d_root = dic['pose3d_root'].reshape(-1, 1, 1).to(device)
    #print(xyz.shape,k.shape)
    uvd = perspectiveProjection(xyz, k).reshape(bs, 21, 3)

    uvd[:,:,-1:]=(uvd[:,:,-1:].clone()-pose3d_root)/scaled
    uvd = (uvd.clone()+transition)*scale

    return uvd

def croppeduvd2xyzAndUnd(uvd,dic,mask=None,wristRoot=False):
    N, device = uvd.shape[0], uvd.device
    #print('uvd.shape',uvd.shape)
    uvd = invertAugmentation(uvd, dic, mask)
    scale = dic['scale'].reshape(-1, 1, 3).to(device)
    transition = dic['transition'].reshape(-1, 1, 3).to(device)
    scaled = dic['scaled'].reshape(-1, 1, 1).to(device)
    k_ = dic['K_'].to(device)
    pose3d_root = dic['pose3d_root'].reshape(-1, 1, 1).to(device)
    if mask is not None:
        scale, transition, k_, scaled, pose3d_root = scale[mask], transition[mask], k_[mask], scaled[mask], pose3d_root[
            mask]
    # pose_gt=pose_gt.cpu()
    # pose_rgb=pose_rgb.cpu()
    uvd = uvd / scale - transition
    uvd[:, :, -1:] *= scaled 
    if(wristRoot):uvd[:, :, -1:]=uvd[:, :, -1:].clone()+(uvd[:,4:5,-1:]-uvd[:,0:1,-1:]).clone()#1.37371
    #if(wristRoot):uvd[:, :, -1:]+=(-uvd[:,0:1,-1:]).clone()#1.37371
    uvd[:, :, -1:] += pose3d_root

    xyz = perspectiveBackProjection(uvd, k_).reshape(N, 21, 3)
    return xyz,uvd

def getEuc(pose_rgb,pose_gt,dic,mask=None):
    bs,device=pose_gt.shape[0],pose_gt.device
    pose_rgb = invertAugmentation(pose_rgb, dic, mask)
    pose_gt = invertAugmentation(pose_gt, dic, mask)
    scale=dic['scale'].reshape(-1,1,3).to(device)
    transition=dic['transition'].reshape(-1,1,3).to(device)
    scaled=dic['scaled'].reshape(-1,1,1).to(device)
    k_=dic['K_'].to(device)
    pose3d_root=dic['pose3d_root'].reshape(-1,1,1).to(device)
    if mask is not None:
        scale,transition,k_,scaled,pose3d_root=scale[mask],transition[mask],k_[mask],scaled[mask],pose3d_root[mask]
    #pose_gt=pose_gt.cpu()
    #pose_rgb=pose_rgb.cpu()
    pose_gt=pose_gt/scale-transition
    pose_rgb=pose_rgb/scale-transition
    pose_gt[:,:,-1:]*=scaled
    pose_gt[:,:,-1:]+=pose3d_root
    pose_rgb[:,:,-1:]*=scaled
    pose_rgb[:,:,-1:]+=pose3d_root

    xyz0 = perspectiveBackProjection(pose_rgb, k_).reshape(bs,21,3)
    xyzgt = perspectiveBackProjection(pose_gt, k_).reshape(bs, 21, 3)

    eucloss=torch.mean(torch.sqrt(torch.sum((xyz0-xyzgt)**2,dim=2)))
 
    return eucloss,xyz0,xyzgt,pose_rgb,pose_gt