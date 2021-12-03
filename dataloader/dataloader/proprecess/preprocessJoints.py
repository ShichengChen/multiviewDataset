import numpy as np
import torch
from cscPy.mano.network.utils import *
RHD2Bighand_skeidx = [0,4,8,12,16,20,3,2,1,7,6,5,11,10,9,15,14,13,19,18,17]
MF2RHD=[0,20,19,18,17,4,3,2,1,8,7,6,5,12,11,10,9,16,15,14,13]
Bighand2MF=[0,2,9,10,11,3,12,13,14,4,15,16,17,5,18,19,20,1,6,7,8]
mcp = [1, 5, 9, 13, 17]
mid = [2, 6, 10, 14, 18]
dip = [3, 7, 11, 15, 19]
tip = [4, 8, 12, 16, 20]
multiple0 = np.array([1.486806, 1.363457, 1.1637201, 1.1142688, 1.1363479]).reshape(-1, 1)
#multiple0 = np.array([1.486806, 1.363457, 1.2, 1.2, 1.2]).reshape(-1, 1)
multiple1 = np.array([1.1343263, 1.1370358, 1.0368003, 1.0731235, 1.2805481]).reshape(-1, 1)
def fadjust(joints_gt,alltip=True,allmcp=True):
    # joints_gt[2], joints_gt[3] = joints_gt[3] + (joints_gt[2] - joints_gt[3]) / 10 * 9, \
    #                              joints_gt[3] + (joints_gt[2] - joints_gt[3]) / 10
    # joints_gt[6], joints_gt[7] = joints_gt[7] + (joints_gt[6] - joints_gt[7]) / 10 * 9, \
    #                              joints_gt[7] + (joints_gt[6] - joints_gt[7]) / 10
    # joints_gt[10], joints_gt[11] = joints_gt[11] + (joints_gt[10] - joints_gt[11]) / 10 * 9, \
    #                                joints_gt[11] + (joints_gt[10] - joints_gt[11]) / 10
    # joints_gt[14], joints_gt[15] = joints_gt[15] + (joints_gt[14] - joints_gt[15]) / 10 * 9, \
    #                                joints_gt[15] + (joints_gt[14] - joints_gt[15]) / 10
    #
    # mdd = np.sqrt(np.sum((joints_gt[dip] - joints_gt[mid]) ** 2, axis=1)).reshape(-1, 1)
    # mmd = np.sqrt(np.sum((joints_gt[mcp] - joints_gt[mid]) ** 2, axis=1)).reshape(-1, 1)
    # mtd = np.sqrt(np.sum((joints_gt[tip] - joints_gt[dip]) ** 2, axis=1)).reshape(-1, 1)
    # d = joints_gt[mcp] - joints_gt[mid]
    # multiple = mdd / mmd
    # joints_gt[mcp] = joints_gt[mid] + d * multiple
    # multiple = mdd / mtd
    # d = joints_gt[tip] - joints_gt[dip]
    # joints_gt[tip] = joints_gt[dip] + d * multiple

    # multiple=np.array([1.2,1.2,1.1637201,1.1142688,1.1363479]).reshape(-1,1)
    # multiple=multiple*(mdd / mmd)
    if allmcp:
        d = joints_gt[mcp] - joints_gt[mid]
        joints_gt[mcp] = joints_gt[mid] + d * multiple0
    # *1.3 7.09
    # *1.2 5.3
    if alltip:
        d = joints_gt[tip] - joints_gt[dip]
        # multiple = multiple * (mdd / mtd)
        joints_gt[tip] = joints_gt[dip] + d * multiple1
    return joints_gt

def invAdjust(joints_gt):
    if(joints_gt.ndim==4):
        d = joints_gt[:,:,mcp] - joints_gt[:,:,mid]
        joints_gt[:,:,mcp] = joints_gt[:,:,mid] + d * (1/multiple0)
        d = joints_gt[:,:,tip] - joints_gt[:,:,dip]
        joints_gt[:,:,tip] = joints_gt[:,:,dip] + d * (1 / multiple1)
    elif (joints_gt.ndim == 3):
        d = joints_gt[:, mcp] - joints_gt[:, mid]
        joints_gt[:, mcp] = joints_gt[:, mid] + d * (1 / multiple0)
        d = joints_gt[:, tip] - joints_gt[:, dip]
        joints_gt[:, tip] = joints_gt[:, dip] + d * (1 / multiple1)
    elif(joints_gt.ndim==2):
        d = joints_gt[mcp] - joints_gt[mid]
        joints_gt[mcp] = joints_gt[mid] + d * (1 / multiple0)
        d = joints_gt[tip] - joints_gt[dip]
        joints_gt[tip] = joints_gt[dip] + d * (1 / multiple1)
    return joints_gt

def preprocessJoints(kp_coord_xyz,adjust=True,usedirect=True,meter2mm=1000):
    '''
    :param kp_coord_xyz: shape=21*3, length=mm
    :return: pose3d_normed, scale, pose3d_root, direct_normed
    '''
    assert (kp_coord_xyz.shape==(21,3))
    '''
    0.032998502 1.486806 1.1343263
    0.031735186 1.363457 1.1370358
    0.028843816 1.1637201 1.0368003
    0.021111924 1.1142688 1.0731235
    0.030788423 1.1363479 1.2805481
    0.038568903
    jidx=[[1,2,3,17],[4,5,6,18],[10,11,12,19],[7,8,9,20],[13,14,15,16]]
    #index,middle,ring,pinky,thumb
    for idxs in jidx:
        a=np.sqrt(np.sum((joints[idxs[0]]-joints[idxs[1]])**2))
        b=np.sqrt(np.sum((joints[idxs[1]]-joints[idxs[2]])**2))
        c=np.sqrt(np.sum((joints[idxs[2]]-joints[idxs[3]])**2))
        print(a,a/b,c/b)
    print(np.sqrt(np.sum((joints[13]-joints[0])**2)))
    '''
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