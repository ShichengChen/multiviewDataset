import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from cscPy.mano.network.utilsSmallFunctions import *
from cscPy.Const.const import *


def getCurvature(joints):
    if(torch.is_tensor(joints)):
        N=joints.shape[0]
        joints=joints.reshape(N,21,3)
        c,n,b,mcps=[],[],[],[7,10,4,1,13]
        for i in mcps:
            b.append(joints[:,i]-joints[:,0])
        for i in range(len(mcps)-1):
            n.append(torch.cross(b[i+1],b[i],dim=1))
        e=[n[0],(n[0]+n[1])/2,(n[1]+n[2])/2,(n[2]+n[3])/2,n[3]]
        for i in range(4):
            cur=torch.sum((e[i+1]-e[i])*(b[i+1]-b[i]),dim=1).reshape(N)/\
                torch.sum((b[i+1]-b[i])*(b[i+1]-b[i]),dim=1).reshape(N)
            c.append(cur.reshape(N,1))
        return torch.cat(c,dim=1).reshape(N,len(mcps)-1)
    else:
        assert False,"not implement"

#manopdx = [1, 2, 3, 17, 4, 5, 6, 18, 7, 8, 9, 20, 10, 11, 12, 19, 13, 14, 15, 16,  4,7,10,  1, 4, 7,10]
#manopdx = [1, 2, 3, 17, 4, 5, 6, 18, 7, 8, 9, 20, 10, 11, 12, 19, 13, 14, 15, 16,  1,7,10, 7]
#manoppx = [0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 10, 11, 12, 0, 13, 14, 15,       1,1, 1, 13,13,13,13]
#manoppx = [0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 10, 11, 12, 0, 13, 14, 15,        4,4, 4,10]
def getBoneLen(joints_gt):

    # manopdx = [1, 2, 3, 17, 4, 5, 6, 18, 7, 8, 9, 20, 10, 11, 12, 19, 13, 14, 15, 16,  1,7,10, 7,13,13,]
    # manoppx = [0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 10, 11, 12, 0, 13, 14, 15,       4,4, 4,10, 4, 1,]
    manopdx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 13, 14, 15, 16,   1, 7, 10, 7, 13, 13, ]
    manoppx = [1, 2, 3,  4, 5, 6,  7, 8, 9,  10, 11, 12, 0,  13, 14, 15,   4, 4, 4, 10, 4, 1, ]
    manopdx += [1, 4, 7, 10, ]
    manoppx += [0, 0, 0, 0, ]

    manopdx = [1, 2, 3, 17, 4, 5, 6, 18,  7, 8, 9, 20,  10,11, 12, 19, 13, 14, 15, 16, 1, 7, 10, 7, 13, 13, ]
    manoppx = [0, 1, 2, 3,  0, 4, 5, 6,   0, 7, 8, 9,   0, 10, 11, 12, 0,  13, 14, 15, 4, 4, 4, 10, 4, 1, ]
    assert len(manopdx)==len(manoppx)
    if (torch.is_tensor(joints_gt)):
        N=joints_gt.shape[0]
        out=torch.zeros([N,len(manopdx)],dtype=joints_gt.dtype,device=joints_gt.device)
        for i in range(len(manoppx)):
            ci = manoppx[i]
            pi = manopdx[i]
            dp = torch.norm(joints_gt[:, ci] - joints_gt[:, pi], dim=1)
            out[:,i]=dp.clone()
        return out

    else:
        # joints_gt=get32fTensor(joints_gt.reshape(21,3))
        # out=[]
        # for i in range(len(manoppx)):
        #     ci = manoppx[i]
        #     pi = manopdx[i]
        #     d=torch.norm(joints_gt[ci] - joints_gt[pi])
        #     out.append(d)
        # return torch.tensor(out,dtype=float32).reshape(len(manopdx)).numpy()
        joints_gt = joints_gt.reshape(21, 3).copy()
        out = []
        for i in range(len(manoppx)):
            ci = manoppx[i]
            pi = manopdx[i]
            d = np.linalg.norm(joints_gt[ci] - joints_gt[pi])
            out.append(d)
        return np.array(out).reshape(len(manopdx))


def getBonePalmMeanStd(joints,bonecoeff=1.6,palmcoeff=2,debug=False):
    if (joints.shape==(21,3)):
        boneLenMean=getBoneLen(joints).reshape(-1)
        curvatureMean=getCurvature(get32fTensor(joints).reshape(1,21,3)).numpy().reshape(-1)
        return get32fTensor(boneLenMean),get32fTensor(np.zeros_like(boneLenMean)),\
               get32fTensor(curvatureMean),get32fTensor(np.zeros_like(curvatureMean))

    out = getBoneLen(get32fTensor(joints))
    out2 = getCurvature(get32fTensor(joints))
    boneLenMean, boneLenStd = torch.mean(out, dim=0, keepdim=True), \
                                        torch.std(out, dim=0, keepdim=True) * bonecoeff
    curvatureMean, curvatureStd = torch.mean(out2, dim=0, keepdim=True), \
                                            torch.std(out2, dim=0, keepdim=True) * palmcoeff
    if (debug):
        loss = torch.mean(torch.max(torch.abs(out - boneLenMean) - boneLenStd, torch.zeros_like(out)))
        loss1 = torch.mean(torch.max(torch.abs(out2 - curvatureMean) - curvatureStd, torch.zeros_like(out2)))
        print('boneLenStd',boneLenStd.reshape(-1)[:16])
        print('gtboneloss,gtcloss', loss, loss1)
    boneLenMean, boneLenStd = boneLenMean.reshape(-1), boneLenStd.reshape(-1)
    #print('init bone len bonestd',boneLenMean, boneLenStd)
    curvatureMean, curvatureStd = curvatureMean.reshape(-1), curvatureStd.reshape(-1)
    return boneLenMean,boneLenStd,curvatureMean,curvatureStd


def getRefJointsFromDataset(joints,palmidx):
    manoidx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
    manopdx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    manoppx = [0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
    # assert type(joints) is np.ndarray
    OriManotempJ = joints[palmidx].reshape(21, 3).copy()
    manotempJ = OriManotempJ.copy()
    for idx in range(len(manoidx)):
        ci = manoidx[idx]
        pi = manopdx[idx]
        ppi = manoppx[idx]
        #dp = np.linalg.norm(OriManotempJ[ci] - OriManotempJ[pi]) + 1e-8
        dp = np.mean(np.linalg.norm(joints[:,ci] - joints[:,pi],axis=1)) + 1e-8
        dm = np.linalg.norm(OriManotempJ[pi] - OriManotempJ[ppi]) + 1e-8
        manotempJ[ci] = manotempJ[pi] + (manotempJ[pi] - manotempJ[ppi]) / dm * dp
    return manotempJ



def getTemplateFrom(boneLen,manoTemplate):
    if(torch.is_tensor(boneLen)):boneLen=boneLen.clone().detach().cpu().numpy()
    boneLen=boneLen.copy().reshape(20)
    boneLen=np.concatenate([[0],boneLen],axis=0).reshape(21)
    OriManotempJ = manoTemplate.reshape(21, 3)
    manotempJ = OriManotempJ.copy()

    boneidxpalm=[1,5,9,13,17]
    manopalm=[1,4,7,10,13]
    for i in range(5):
        ci=manopalm[i]
        dm=np.linalg.norm(OriManotempJ[ci] - OriManotempJ[0]) + 1e-8
        manotempJ[ci]=manotempJ[0]+(OriManotempJ[ci]-OriManotempJ[0])/dm*boneLen[boneidxpalm[i]]


    manoidx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
    manopdx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    manoppx = [0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
    boneidxfinger=[2,3,4,6,7,8,10,11,12,14,15,16,18,19,20]

    for idx in range(len(manoidx)):
        ci = manoidx[idx]
        pi = manopdx[idx]
        ppi = manoppx[idx]
        dp = boneLen[boneidxfinger[idx]] + 1e-8
        dm = np.linalg.norm(OriManotempJ[pi] - OriManotempJ[ppi]) + 1e-8
        manotempJ[ci] = manotempJ[pi] + (manotempJ[pi] - manotempJ[ppi]) / dm * dp
    return manotempJ



def getRefJoints(joint_gt):
    manoidx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
    manopdx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    manoppx = [0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
    if(torch.is_tensor(joint_gt)):
        N=joint_gt.shape[0]
        OriManotempJ = torch.tensor(joint_gt.reshape(N, 21, 3),dtype=torch.float32,device=joint_gt.device)
        manotempJ = OriManotempJ.clone()
        for idx in range(len(manoidx)):
            ci = manoidx[idx]
            pi = manopdx[idx]
            ppi = manoppx[idx]
            dp = torch.norm(OriManotempJ[:, ci] - OriManotempJ[:, pi], dim=-1, keepdim=True)+1e-8
            dm = torch.norm(OriManotempJ[:, pi] - OriManotempJ[:, ppi], dim=-1, keepdim=True)+1e-8
            manotempJ[:, ci] = manotempJ[:, pi] + (manotempJ[:, pi] - manotempJ[:, ppi]) / dm * dp
        return manotempJ
    else:
        OriManotempJ = joint_gt.reshape(21, 3)
        manotempJ = OriManotempJ.copy()
        for idx in range(len(manoidx)):
            ci = manoidx[idx]
            pi = manopdx[idx]
            ppi = manoppx[idx]
            dp = np.linalg.norm(OriManotempJ[ci] - OriManotempJ[pi]) + 1e-8
            dm = np.linalg.norm(OriManotempJ[pi] - OriManotempJ[ppi]) + 1e-8
            manotempJ[ci] = manotempJ[pi] + (manotempJ[pi] - manotempJ[ppi]) / dm * dp
        return manotempJ



def disPoint2Plane(points,planeNorm,planeD):
    N = points.shape[0]
    return (torch.sum((points.reshape(N,3) * planeNorm.reshape(N,3)).reshape(N, 3), dim=1, keepdim=True)
           + planeD.reshape(N,1)).reshape(N, 1)

def projectPoint2Plane(points,planeNorm,planeD,relaxValue=0):
    N=points.shape[0]
    points=points.reshape(N,3)
    planeNorm=unit_vector(planeNorm)
    #https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d
    dis=disPoint2Plane(points,planeNorm,planeD).reshape(N,1)
    #plane: ax+by+cz+d=0
    #norm: (a,b,c)
    #dis=norm*point+d
    if(relaxValue==0):
        pass
    else:
        mask=dis<0
        dis=torch.max(torch.abs(dis) - relaxValue, torch.zeros_like(dis))
        dis[mask]*=-1
    projectedPoint = (points - dis*planeNorm.reshape(N,3)).reshape(N, 3)
    #ans=point-dist*norm
    return torch.abs(epsilon+dis),projectedPoint



def getBatchTransitionMatrix3D(x):
    bs=x.shape[0]
    x=x.reshape(bs,3)
    device=x.device
    a=torch.eye(4, dtype=torch.float32, device=device).reshape(1, 4, 4).repeat(bs, 1, 1)
    a[:,:-1,-1]=x
    return a

def getTransitionMatrix3D(x=0,y=0,z=0):
    return np.array([[1, 0,0,x],[0, 1,0,y],[0, 0,1,z],[0,0,0,1]])

def AxisRotMat(angles,rotation_axis):
    ''' rotation matrix from rotation around axis
        see https://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
        #https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
         [[cos+self.xx*(1-cos), self.xy*(1-cos)-self.z*sin, self.xz*(1-cos)+self.y*sin, 0.0],
          [self.xy*(1-cos)+self.z*sin, cos+self.yy*(1-cos), self.yz*(1-cos)-self.x*sin, 0.0],
          [self.xz*(1-cos)-self.y*sin, self.yz*(1-cos)+self.x*sin, cos+self.zz*(1-cos), 0.0],
          [0.0, 0.0, 0.0, 1.0]]
    '''
    x,y,z=rotation_axis
    xx,xy,xz,yy,yz,zz=x*x,x*y,x*z,y*y,y*z,z*z
    c = np.cos(angles)
    s = np.sin(angles)
    i = 1 - c
    rot_mats=np.eye(4).astype(np.float32)
    rot_mats[0,0] =  xx * i + c
    rot_mats[0,1] =  xy * i -  z * s
    rot_mats[0,2] =  xz * i +  y * s

    rot_mats[1,0] =  xy * i +  z * s
    rot_mats[1,1] =  yy * i + c
    rot_mats[1,2] =  yz * i -  x * s

    rot_mats[2,0] =  xz * i -  y * s
    rot_mats[2,1] =  yz * i +  x * s
    rot_mats[2,2] =  zz * i + c
    rot_mats[3,3]=1
    return rot_mats

def rotation_matrix(axis:torch.Tensor, theta:torch.Tensor)->torch.Tensor:
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    N=max(theta.shape[0],axis.shape[0])
    assert len(axis.shape)==2
    axis = unit_vector(axis).reshape(N,3)
    a = torch.cos(theta / 2.0).reshape(N)
    curbcd = -axis * torch.sin(theta / 2.0).reshape(N,1)
    b, c, d = curbcd[:,0],curbcd[:,1],curbcd[:,2]
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    out=torch.zeros([N,3,3],dtype=theta.dtype,device=theta.device)
    out[:,0,0]=aa + bb - cc - dd
    out[:,0,1]=2 * (bc + ad)
    out[:,0,2]=2 * (bd - ac)
    out[:,1,0]=2 * (bc - ad)
    out[:,1,1]=aa + cc - bb - dd
    out[:,1,2]=2 * (cd + ab)
    out[:,2,0]=2 * (bd + ac)
    out[:,2,1]=2 * (cd - ab)
    out[:,2,2]=aa + dd - bb - cc
    return out
    # return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
    #                  [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
    #                  [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def getRotationMatrix3D(thetax,thetay,thetaz):
    rx=AxisRotMat(thetax,[1,0,0])
    ry=AxisRotMat(thetay,[0,1,0])
    rz=AxisRotMat(thetaz,[0,0,1])
    return rx@ry@rz

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    if (torch.is_tensor(v1_u)):
        bs=v1_u.shape[0]
        v1_u=v1_u.reshape(bs,3)
        return torch.acos(torch.clamp(torch.sum(v1_u*v2_u,dim=1), -1.0, 1.0))
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotate2joint(wrist_trans,local_trans,template,parent):
    device = wrist_trans.device
    Rs = torch.cat([wrist_trans, local_trans], dim=1)
    N = Rs.shape[0]
    root_rotation = Rs[:, 0, :, :]
    Js = torch.unsqueeze(template, -1)

    def make_A(R, t):
        R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
        ones_homo = Variable(torch.ones(N, 1, 1))
        ones_homo = ones_homo.to(device)
        t_homo = torch.cat([t, ones_homo], dim=1)
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]
    newjs = template.clone()

    newjsones = torch.ones([newjs.shape[0], 21, 1]).to(device)
    newjs = torch.cat([newjs, newjsones], dim=2).reshape(N, 21, 4, 1)
    orijs = newjs.clone()
    transidx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
    cpidx = [1, 4, 7, 10, 13]

    for i in range(5):
        a = minusHomoVectors(orijs[:, cpidx[i]], orijs[:, 0]).reshape(N,4,1)
        newjs[:, cpidx[i]] = (A0 @ a)

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)

        a = minusHomoVectors(orijs[:, transidx[i - 1]], orijs[:, i])
        newjs[:, transidx[i - 1]] = (res_here @ a).reshape(N,4,1)
        results.append(res_here)

    return newjs[:,:,:-1].reshape(N,21,3)



def planeAlignment(temp0,temp1,targetd0,targetd1):
    N=temp0.shape[0]
    #print('targetd0',targetd0)
    r = getRotationBetweenTwoVector(temp0, targetd0)
    pl0 = r @ temp1.reshape(N, 3, 1)
    pl0 = pl0.reshape(N, 3)
    pl1 = targetd1
    pl1 = pl1.reshape(N, 3)
    r2 = getRotationBetweenTwoVector(pl0, pl1)
    return r2@r

# def wristRotTorch(tempJ,joint_gt):
#     isnumpy=False
#     if(not torch.is_tensor(tempJ)):
#         isnumpy=True
#         tempJ,joint_gt=get32fTensor(tempJ).reshape(1,21,3),get32fTensor(joint_gt).reshape(1,21,3)
#     N=tempJ.shape[0]
#     ##assume norm of palm is z direction
#     ##wirst to mmcp is y direction
#     z0 = unit_vector(torch.cross(tempJ[:, 4] - tempJ[:, 0], tempJ[:, 7] - tempJ[:, 4], dim=1))
#     z1 = unit_vector(torch.cross(joint_gt[:, 4] - joint_gt[:, 0], joint_gt[:, 7] - joint_gt[:, 4], dim=1))
#     y0 = (tempJ[:, 4] - tempJ[:, 0]).reshape(N, 3, 1)
#     y1 = (joint_gt[:, 4] - joint_gt[:, 0])
#     #print('z0, y0, z1, zy',z0,y0,z1,y1)
#     r=planeAlignment(z0,y0,z1,y1)
#     if isnumpy:return r.reshape(3,3).numpy()
#     return r

def wristRotTorch(tempJ,joint_gt):
    assert torch.is_tensor(tempJ)
    plamsIdx=[0,1,4,7,10,13]
    N, n, d = tempJ.shape[0], len(plamsIdx), tempJ.shape[2]
    a=tempJ[:,plamsIdx].clone().reshape(N,n,3)
    b=joint_gt[:,plamsIdx].clone().reshape(N,n,3)
    aave = torch.sum(a, dim=1, keepdim=True) / n
    bave = torch.sum(b, dim=1, keepdim=True) / n
    x = (a - aave.reshape(N,1,d)).reshape(N, n, d)
    y = (b - bave.reshape(N,1,d)).reshape(N, n, d)
    r = svdForRotationWithoutW(x, y)
    t = bave - aave @ r
    return r,t.reshape(N,1,d)




def svdForRotation(a,weight,b):
    N,n,d=a.shape[0],a.shape[1],a.shape[2]
    #weight[(0<weight) & (weight<1)]=1
    trw = torch.diag(weight).repeat(N, 1, 1).reshape(N, n, n)
    w = (a.permute(0, 2, 1)) @ trw @ b
    w = w.cpu()
    u, _, v = w.svd()
    s = v.reshape(N, 3, 3) @ (u.reshape(N, 3, 3).permute(0, 2, 1))
    s = torch.det(s)
    s = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, s]])
    r = v.reshape(N, 3, 3) @ s @ (u.reshape(N, 3, 3).permute(0, 2, 1))
    r = r.to(a.device).reshape(N,3,3)
    return r
def svdForRotationWithoutW(a,b):
    N,n,d=a.shape[0],a.shape[1],a.shape[2]
    w = (a.permute(0, 2, 1)) @ b
    w = w.cpu()
    u, _, v = w.svd()
    s = v.reshape(N, d, d) @ (u.reshape(N, d, d).permute(0, 2, 1))
    s = torch.det(s)
    #print(N,n,d)
    I = torch.eye(d,d).reshape(1,d,d).repeat(N,1,1).float()
    I[:,-1,-1]=s.clone()
    #print(I.shape)
    #s = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, s]])
    r = v.reshape(N, d, d) @ I @ (u.reshape(N, d, d).permute(0, 2, 1))
    r = r.to(a.device).reshape(N,3,3)
    return r

def getPlaneFrom4Points(joints: torch.Tensor)->(torch.Tensor,torch.Tensor):
    N,n,d=joints.shape
    assert n==4
    from itertools import combinations
    subsets = list(combinations([0,1,2,3], 3))
    subsets = [[0,2,3],[0,1,3]]
    #subsets = [[0,1,2],[0,1,3]]
    vlist = []
    for subset in subsets:
        v0 = joints[:, subset[0]] - joints[:, subset[1]]
        v1 = joints[:, subset[1]] - joints[:, subset[2]]
        vh = torch.cross(v0, v1, dim=1)
        vlist.append(vh.reshape(1, N, 3))
    vh = unit_vector(torch.mean(torch.cat(vlist, dim=0), dim=0).reshape(N, 3)).reshape(N,1,3)
    #vh = torch.mean(torch.cat(vlist, dim=0), dim=0).reshape(N, 1,3)
    subj = joints.reshape(N, 4, 3)
    vd = torch.mean(-torch.sum(subj * vh, dim=2), dim=1).reshape(N, 1)
    return vh,vd


def getFingerStdDir(joints:torch.Tensor,idx:int)->torch.Tensor:
    #N = joints.shape[0]
    #if(idx==0):return unit_vector(joints[:,1]-joints[:,4])
    # if(0<=idx<=2):return unit_vector(joints[:,1]-joints[:,10])
    # elif(idx==3):return unit_vector(joints[:,4]-joints[:,7])
    # elif idx==4:return unit_vector(joints[:,13]-joints[:,1])

    if (idx <= 1):
        return unit_vector(joints[:, 1] - joints[:, 10])
    if (3 >= idx >= 2):
        return unit_vector(joints[:, 4] - joints[:, 7])
    elif idx == 4:
        return unit_vector(joints[:, 13] - joints[:, 1])
    # normidx = [-1, -1, -1, -1, 0]  # index,middle,ringy,pinky,thumb
    # mcpidx = [1, 4, 10, 7, 13]
    # palmNorm = getPalmNormByIndex(joints, normidx[idx]).reshape(N, 3)  # palm up
    # vecWristMcp = unit_vector(joints[:, mcpidx[idx]] - joints[:, 0]).reshape(N, 3)  # wirst to mmcp
    # stdFingerNorm = unit_vector(torch.cross(vecWristMcp, palmNorm, dim=1))  # direction from pmcp to imcp
    # return stdFingerNorm

def getPalmNormByIndex(joints: torch.Tensor, idx: int) -> torch.Tensor:
    if (idx == -1): return getPalmNorm(joints)
    assert 0 <= idx <= 4, "bad index"
    #c = [(13, 1), (1, 4), (4, 10), (10, 7)] (18/9/21 version) 1223normidx
    c = [(1, 4), (4, 10), (4, 10), (10, 7), (13, 1)]
    #c = [(1, 10), (1, 10), (1, 10), (10, 7), (13, 1)]
    #c = [(1, 10), (1, 10), (4, 7), (4, 7),(13, 1),] #new
    return unit_vector(
        torch.cross(joints[:, c[idx][0]] - joints[:, 0], joints[:, c[idx][1]] - joints[:, c[idx][0]], dim=1))

def getPalmNorm(joints: torch.Tensor, ) -> torch.Tensor:
    palmNorm = unit_vector(torch.cross(joints[:, 4] - joints[:, 0], joints[:, 7] - joints[:, 4], dim=1))
    return palmNorm

def getRotationBetweenTwoVector(a,b):
    if(torch.is_tensor(a)):
        #print('a,b',a,b)
        device=a.device
        bs = a.shape[0]
        a = unit_vector(a)
        b = unit_vector(b)
        a=a.reshape(bs,3)
        G=torch.eye(3,dtype=torch.float32,device=device).reshape(1,3,3).repeat(bs,1,1).reshape(bs,3,3)
        G[:, 0, 0] = torch.sum(a * b, dim=1)
        G[:, 0, 1] = -torch.norm(torch.cross(a,b,dim=1), dim=1)
        G[:, 1, 0] = torch.norm(torch.cross(a,b,dim=1), dim=1)
        G[:, 1, 1] = torch.sum(a * b, dim=1)
        u=a.clone()
        v=b-torch.sum(a*b,dim=1,keepdim=True)*a
        v = unit_vector(v)
        F = torch.zeros([bs,3,3],dtype=torch.float32,device=device)
        F[:,:, 0], F[:,:, 1], F[:,:, 2] = u, v, unit_vector(torch.cross(b, a, dim=1))

        f = F.cpu()
        #print('f',f)
        #print(np.linalg.matrix_rank(f))
        rf = (torch.sum(torch.svd(f)[1]>1e-4,dim=1) == 3)
        if(rf.device!=device):rf=rf.to(device)
        R = torch.eye(3, dtype=torch.float32, device=device).reshape(1, 3, 3).repeat(bs, 1, 1).reshape(bs, 3, 3)
        R[rf] = F[rf] @ G[rf] @ torch.inverse(F[rf])
        return R
    else:
        a=unit_vector(a).copy()
        b=unit_vector(b).copy()
        if (np.linalg.norm(a - b) < 1e-5): return np.eye(3)
        G=np.array([[np.dot(a,b),- np.linalg.norm(np.cross(a,b)),0],[ np.linalg.norm(np.cross(a,b)),np.dot(a,b),0],[0,0,1]])
        u=a.copy()
        v=b-(np.dot(a,b))*a
        v=unit_vector(v)
        F=np.zeros([3,3],dtype=np.float64)
        F[:,0],F[:,1],F[:,2]=u,v,unit_vector(np.cross(b,a))
        R=F@G@np.linalg.inv(F)
        return R



if __name__ == "__main__":
    print(projectPoint2Plane(get32fTensor(torch.tensor([[0,0,0]])),
                       get32fTensor(torch.tensor([[0.7071,0.7071,0]])),
                       get32fTensor(torch.tensor([[-0.7071]])))
          )#(tensor([[0.7071]]), tensor([[0.5000, 0.5000, 0.0000]]))

    print(projectPoint2Plane(get32fTensor(torch.tensor([[1, 1, 0]])),
                             get32fTensor(torch.tensor([[0.7071, 0.7071, 0]])),
                             get32fTensor(torch.tensor([[-0.7071]])))
          )#(tensor([[0.7071]]), tensor([[0.5000, 0.5000, 0.0000]]))

































































'''
the following code is useless now
'''


def lstsqForR(a,weight,b):
    #here R maybe not real r
    N=a.shape[0]
    r=torch.zeros((N,3,3),dtype=a.dtype,device=a.device)
    x,y=a[:,weight>0],b[:,weight>0]
    for i in range(N):
        X, _ = torch.lstsq(x[i], y[i])
        r[i]=X[:3,:3].clone()
    return r
def icp(coords, coords_ref, n_iter):
    """
    Iterative Closest Point
    """
    for t in range(n_iter):
        cdist = torch.cdist(coords - coords.mean(dim=0),
                            coords_ref - coords_ref.mean(dim=0))
        mindists, argmins = torch.min(cdist, dim=1)
        print(argmins)
        print(coords_ref[argmins].shape)
        print(coords_ref[argmins])
        X, _ = torch.lstsq(coords_ref[argmins].contiguous(), coords)
        coords = coords.mm(X[:3])
        rmsd = torch.sqrt((X[3:]**2).sum(dim=1).mean())
    return coords

def ICP(a,b,mask,root):
    N, n, d = a.shape[0], a.shape[1], a.shape[2]

    assert (N==1)
    mask=mask.reshape(n)
    mask[mask>0]=1
    mask=mask.long()
    root=root.reshape(N,1,d)
    rt=torch.eye(d+1).reshape(1,d+1,d+1).repeat(N,1,1)
    a,b=a[:,mask],b[:,mask]
    print(b.shape)
    a,b=a.clone()-root,b.clone()-root
    print(-1, 'icp error', torch.mean(torch.sqrt(torch.sum((b - a) ** 2, dim=2))))
    for iter in range(10):
        cdist = torch.cdist(a-a.mean(dim=1,keepdim=True),b-b.mean(dim=1,keepdim=True))
        mindists, argmins = torch.min(cdist, dim=2)
        print(argmins)
        #print(argmins)
        # bat = torch.arange(N).reshape(N, 1).repeat(1, 778).reshape(-1)
        # print(argmins.shape)
        # print(b[:,argmins.reshape(-1)].shape)
        print(b.shape)
        bb=b[:,argmins.reshape(-1)].contiguous()
        r,t=getRotationBetweenTwoMeshBone(a.clone(),bb.clone(),None,rt=True,ICP=True)
        crt=torch.eye(d+1).reshape(1,d+1,d+1).repeat(N,1,1)
        #print(r.shape,t.shape)
        crt[:,:d,:d]=r ###omit transition here
        #print(a.shape,r.shape,t.shape)
        a=(r@a.reshape(N,n,d,1)).reshape(N,n,d)
        print(iter,'icp error',torch.mean(torch.sqrt(torch.sum((b-a)**2,dim=2))))
        rt=crt@rt
    return rt[:,:d,:d],rt[:,d:,:d]

def getRotationBetweenTwoMeshBone(a,b,weight,rt=False,ICP=False):
    N,n,d=a.shape[0],a.shape[1],a.shape[2]
    mask = weight.reshape(n).clone()
    a[:, mask<0.2]=0
    b[:, mask<0.2]=0
    if(ICP==True):
        usepytorch3d=True
        # mask = weight.reshape(n).clone()
        # mask[mask > 0.2] = 1
        # mask = mask.bool()
        print(torch.sum(mask>0.5))
        a, b = a[:, mask>0.5].clone(), b[:, mask>0.5].clone()
        print(a,b)
        if(usepytorch3d):
            from pytorch3d.ops import iterative_closest_point
            from pytorch3d.transforms import matrix_to_euler_angles
            icpout = iterative_closest_point(a, b)
            r,t=icpout.RTs.R.permute(0,2,1),icpout.RTs.T
            print(r,t,icpout.rmse)
            print(matrix_to_euler_angles(r,'XYZ'))
            return r, t


        # aave,bave=torch.mean(a,dim=1,keepdim=True),torch.mean(b,dim=1,keepdim=True)
        # x = (a - aave).reshape(N, n, d)
        # y = (b - bave).reshape(N, n, d)
        # r = svdForRotationWithoutW(x, y)
        # t = bave - aave @ r

    weight=weight.reshape(n)
    if(rt==False):
        assert False,"unknown bug here"
        r=svdForRotation(a,weight,b)
        return r
    else:
        aave=(torch.sum(a*weight.reshape(1,n,1),dim=1).reshape(N,d)/torch.sum(weight).reshape(1,1)).reshape(N,1,d)
        bave=(torch.sum(b*weight.reshape(1,n,1),dim=1).reshape(N,d)/torch.sum(weight).reshape(1,1)).reshape(N,1,d)
        x=(a-aave).reshape(N,n,d)
        y=(b-bave).reshape(N,n,d)
        r = svdForRotation(x, weight, y)
        #r = lstsqForR(x,weight,y)
        t = bave - aave @ r
        print('corresponding_points_alignment',r,t)
        from pytorch3d.transforms import matrix_to_euler_angles
        print(matrix_to_euler_angles(r,'XYZ'))
        #aa = (r @ a.reshape(N, 778, 3, 1) + t.reshape(N, 1, 3, 1)).reshape(N, 778, 3)
        '''
        curweight=torch.sum(weight_coeff[bonelist[idx]],dim=0).reshape(N,778,1,1)
        curidx=(torch.sum(weight_coeff[bonelist[idx]],dim=0).reshape(N,778)>0.7)
        tempV[curidx] = (r.reshape(N, 3, 3) @ (tempV.reshape(N, 778, 3, 1)*curweight))[:, :, :, 0][curidx]
        '''
        #print('svd error', torch.mean(torch.sqrt(torch.sum((b - aa) ** 2, dim=2))))
        return r,t


def getHardWeightCoeff(weight):
    weight_coeff = weight[0].permute(1, 0).reshape(16, 778)
    weight_idx = torch.argmax(weight_coeff, dim=0)
    weight_coeff_hard = torch.zeros_like(weight_coeff, device=weight_coeff.device)
    for i in range(weight_coeff.shape[0]):
        weight_coeff_hard[i][weight_idx == i] = 1
    assert torch.sum(torch.abs(torch.sum(weight_coeff_hard, dim=0) - 1) < 0.001), "invalid weight coeff hard"
    return torch.tensor(weight_coeff_hard)
def getOriWeightCoeff(weight):
    return weight[0].permute(1, 0).reshape(16, 778)
def getUsefulWeightCoeff(weight):
    weight_coeff = weight[0].permute(1, 0).reshape(16, 778)
    for i in range(16):
        weight_coeff[i][weight_coeff[i]<0.8]=0
    return weight_coeff
def getPortionWeightCoeff(weight):
    weight_coeff = weight[0].permute(1, 0).reshape(16, 778)
    for i in range(16):
        l=torch.sum(weight_coeff[i]>0)//4
        arg=torch.argsort(weight_coeff[i])
        weight_coeff[i][arg[:-l]]=0
    return weight_coeff
