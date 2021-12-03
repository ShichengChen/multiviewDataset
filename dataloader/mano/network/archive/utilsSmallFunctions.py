import numpy as np
import torch
from cscPy.Const.const import *

def minusHomoVectors(v0, v1):
    v = v0 - v1
    if (v.shape[-1] == 1):
        v[..., -1, 0] = 1
    else:
        v[..., -1] = 1
    return v

def getRtMatrix2D(theta,x=0,y=0):
    return np.array([[np.cos(theta), -np.sin(theta),x],
                     [np.sin(theta), np.cos(theta),y],
                     [0,0,1]])

def comebineRt3D(r,t):
    return np.array([[r[0,0], r[0,1],r[0,2],t[0]],
                     [r[1,0], r[1,1],r[1,2],t[1]],
                     [r[2,0], r[2,1],r[2,2],t[2]],
                     [0,0,0,1]])


def getBatchEpe(a,b):
    return torch.mean(torch.sqrt(torch.sum((a-b)**2,dim=2)))

def batchrtMM(rt,v):
    N=v.shape[0]
    #print(rt.shape)
    return (rt[:,:3,:3].reshape(N,3,3)@v.reshape(N,3,1)).reshape(N,3)+\
           rt[:,:-1,-1].reshape(N,3)

def getTransitionMatrix2D(x=0,y=0):
    return np.array([[1, 0,x],[0, 1,y],[0,0,1]])

def getInhomogeneousLine(lines:np.ndarray):
    l=lines.copy()
    l=l.reshape(-1,l.shape[-1])[:,:-1]
    return l

def getHomo3D(x):
    if(torch.is_tensor(x)):
        if(x.shape[-1]==4):return x
        if(x.shape[-1]==1 and x.shape[-2]==4):return x
        if(x.shape[-1]==1 and x.shape[-2]==3):
            return torch.cat([x, torch.ones([*(x.shape[:-2])] + [1,1], dtype=torch.float32,device=x.device)], dim=-2)
        if(x.shape[-1]==3):
            return torch.cat([x, torch.ones([*(x.shape[:-1])] + [1], dtype=torch.float32,device=x.device)], dim=-1)
    if(x.shape[-1]==3):
        return np.concatenate([x,np.ones([*(x.shape[:-1])]+[1],dtype=np.float64,device=x.device)],axis=-1)
    return x


def get32fTensor(a)->torch.Tensor:
    if(torch.is_tensor(a)):
        return a.float()
    return torch.tensor(a,dtype=torch.float32)
def getBatch(a):
    return get32fTensor(a).unsqueeze(dim=0)

def unit_vector(vec):
    if(torch.is_tensor(vec)):
        bs=vec.shape[0]
        vec=vec.reshape(bs,3)
        return vec / (torch.norm(vec,dim=1,keepdim=True)+1e-8)
    return vec / (np.linalg.norm(vec)+1e-8)

def vector_length(vec):
    assert torch.is_tensor(vec),"only support torch"
    return torch.sqrt(1e-8+torch.sum(vec**2,dim=-1))


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    if (torch.is_tensor(v1_u)):
        bs=v1_u.shape[0]
        v1_u=v1_u.reshape(bs,3)
        return torch.acos(torch.clamp(torch.sum(v1_u*v2_u,dim=1), -1.0, 1.0))
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def vector_dis(v0,v1):
    return float(np.sqrt((v0[0]-v1[0])**2+(v0[1]-v1[1])**2+(v0[2]-v1[2])**2))

def euDist(v0,v1):
    return torch.sqrt(epsilon*1e-3+torch.sum((v0-v1)**2,dim=-1))

def perspectiveBackProjection(xyz_point, K_):
    #k_is inverse of k
    if (not torch.is_tensor(xyz_point)):
        xyz_point=torch.tensor(xyz_point,dtype=torch.float32)
        K_=torch.tensor(K_,dtype=torch.float32)
    out = xyz_point.clone()
    out[..., :-1] = xyz_point[..., :-1] * xyz_point[..., -1:]
    xyz = (K_ @ out.unsqueeze(-1))[...,0]
    return xyz


def perspectiveProjection(xyz_point, camera):
    assert(camera.shape[-1]==3==camera.shape[-2])
    if (torch.is_tensor(xyz_point)):
        xyz=xyz_point.unsqueeze(-1)
        uvd = (camera @ xyz)[..., 0]
        out=uvd.clone()
        out[..., :-1] = uvd[..., :-1] / (uvd[..., -1:].clone()+epsilon)
        return out
    else:
        xyz=np.expand_dims(xyz_point, axis=-1)
        uvd = (camera @ xyz)[..., :, 0]
        uvd[..., :-1] /= uvd[..., -1:].copy()
        return uvd

