import pickle

import torch.nn as nn

from cscPy.mano.network.utils import *
from cscPy.mano.network.utilsSmallFunctions import *
from cscPy.mano.network.Const import boneSpace
from cscPy.Const.const import epsilon
from cscPy.Const.const import *

class BiomechanicalLayer(nn.Module):
    def __init__(self, fingerPlaneLoss=False,fingerFlexLoss=False,fingerAbductionLoss=False,fingerPlaneRotLoss=False,
                 planeFlexRelax=0.95,planeRotRelax=1,boneLenLoss=False,curvatureLoss=False):
        super(BiomechanicalLayer, self).__init__()
        ##only works for right hand!!!
        self.fingerPlaneLoss=fingerPlaneLoss
        self.fingerFlexLoss=fingerFlexLoss
        self.fingerAbductionLoss=fingerAbductionLoss
        self.fingerPlaneRotLoss=fingerPlaneRotLoss
        self.nocheckScale=True
        self.planeFlexRelax=planeFlexRelax
        self.planeRotRelax=planeRotRelax
        self.boneLenLoss=boneLenLoss
        self.curvatureLoss=curvatureLoss

    def forward(self,joints: torch.Tensor,scale:torch.Tensor, info:dict):
        N=joints.shape[0]
        if(scale is None):bonelen=torch.ones(N,dtype=joints.dtype,device=joints.device)
        else: bonelen=scale.reshape(N)
        loss,disEucloss={'p':0,'pr':0,'f':0,'ab':0,'bl':0,'c':0},{'p':0,'pr':0,'f':0,'ab':0,'bl':0,'c':0}
        loss['p'],disEucloss['p'], joint_gt = self.restrainFingerDirectly(joints, bonelen)
        if(self.boneLenLoss):
            loss['bl'],disEucloss['bl']=self.getBoneLenLoss(joints,bonelen,info)
        if(self.curvatureLoss):
            loss['c'],disEucloss['c']=self.getCurvatureLoss(joints,bonelen,info)
        if(self.fingerPlaneLoss):
            pass
        if(self.fingerPlaneRotLoss):
            loss['pr'],disEucloss['pr']=self.fingerPlaneRotationLoss(joint_gt.clone(),bonelen)
        if(self.fingerFlexLoss):
            loss['f'],disEucloss['f']=self.restrainFlexAngle(joint_gt.clone(),bonelen)
            #print("flex loss",loss)
        if(self.fingerAbductionLoss):
            loss['ab'],disEucloss['ab']=self.restrainAbductionAngle(joint_gt.clone(),bonelen)
        return loss,disEucloss

    def getBoneLenLoss(self,joint_gt:torch.Tensor,bonelen: torch.Tensor,info:dict):
        N=joint_gt.shape[0]
        out=getBoneLen(joint_gt.reshape(N,21,3)*bonelen.reshape(N,1,1)*1000)
        bmean=info['boneLenMean'].to(joint_gt.device)
        bstd=info['boneLenStd'].to(joint_gt.device)
        loss=torch.mean(torch.max(torch.abs(out-bmean)-bstd,torch.zeros_like(out)))
        return loss,loss

    def getCurvatureLoss(self,joint_gt:torch.Tensor,bonelen: torch.Tensor,info:dict):
        N=joint_gt.shape[0]
        out=getCurvature(joint_gt.reshape(N,21,3)*bonelen.reshape(N,1,1)*1000)
        bmean=info['curvatureMean'].to(joint_gt.device)
        bstd=info['curvatureStd'].to(joint_gt.device)
        loss=torch.mean(torch.max(torch.abs(out-bmean)-bstd,torch.zeros_like(out)))
        return loss,loss


    def fingerPlaneRotationLoss(self,joint_gt:torch.Tensor,bonelen: torch.Tensor,)\
            ->(torch.Tensor,torch.Tensor,torch.Tensor):
        N = joint_gt.shape[0]
        normidx = [1, 2, 2, 3]
        jidx = [[0, 1, 2, 3, 17], [0, 4, 5, 6, 18], [0, 10, 11, 12, 19], [0, 7, 8, 9, 20]]
        angleP = torch.tensor([np.pi / 6],device=joint_gt.device, dtype=joint_gt.dtype)
        angleM = torch.tensor([np.pi / 2],device=joint_gt.device, dtype=joint_gt.dtype)
        loss,euc=torch.zeros_like(bonelen),torch.zeros_like(bonelen)
        for idx,finger in enumerate(jidx):
            #v0 = unit_vector(joint_gt[:, finger[1]] - joint_gt[:, finger[0]])
            v1 = unit_vector(joint_gt[:, finger[2]] - joint_gt[:, finger[1]])
            v2 = unit_vector(joint_gt[:, finger[3]] - joint_gt[:, finger[2]])
            #v3 = unit_vector(joint_gt[:, finger[4]] - joint_gt[:, finger[3]])

            #mask=(torch.abs(torch.sum(v1 * v2, dim=1))<self.planerelax)
            mask=(torch.abs(torch.sum(v1 * v2, dim=1))<self.planeRotRelax)

            palmNorm = unit_vector(self.getPalmNormByIndex(joint_gt, normidx[idx]).reshape(N, 3))  # palm up
            dis=euDist(joint_gt[:, finger[1]],joint_gt[:, finger[2]]).reshape(N)
            a = unit_vector(torch.cross(v1, v2, dim=1)).reshape(N,3)
            #b = unit_vector(torch.cross(v2, v3, dim=1)).reshape(N,3)

            angle = torch.acos(torch.clamp(torch.sum(a * palmNorm, dim=1), -1 + epsilon, 1 - epsilon)).reshape(-1)

            cur=torch.max(torch.abs(angle-angleM+epsilon)-angleP,torch.zeros_like(angle))*dis
            cur[mask==False]=0
            loss += cur
            #print('cur',cur*500)
            euc+=cur*bonelen
                #print(finger,angle/3.14*180)
        return torch.mean(loss)/4,torch.mean(euc)/4


    def restrainFingerDirectly(self, joint_gt: torch.Tensor,bonelen: torch.Tensor,)\
            ->(torch.Tensor,torch.Tensor,torch.Tensor):
        N = joint_gt.shape[0]
        newjoints_gt=joint_gt.clone()
        jidx = [[1, 2, 3, 17], [4, 5, 6, 18], [10, 11, 12, 19], [7, 8, 9, 20], [13, 14, 15, 16]]

        loss,euc=0,0
        for finger in jidx:
            vh,vd=getPlaneFrom4Points(joint_gt[:,finger].clone())
            for idx in range(4):
                cur, newjoints_gt[:, finger[idx]] = \
                    projectPoint2Plane(joint_gt[:, finger[idx]], vh, vd)
                # distance[:, finger[idx]]
                loss+=cur.reshape(N)
                euc+=cur.reshape(N)*bonelen
        return torch.mean(loss)/21,torch.mean(euc)/21, newjoints_gt

    #n0=wrist tmcp imcp
    #n1=wrist imcp mmcp
    #n2=wrist mmcp rmcp
    #n3=wrist rmcp pmcp
    def getPalmNormByIndex(self,joint_gt: torch.Tensor,idx:int) -> torch.Tensor:
        if(idx==-1):return self.getPalmNorm(joint_gt)
        assert 0<=idx<4,"bad index"
        c=[(13,1),(1,4),(4,10),(10,7)]
        return unit_vector(torch.cross(joint_gt[:, c[idx][0]] - joint_gt[:, 0], joint_gt[:, c[idx][1]] - joint_gt[:, c[idx][0]], dim=1))


    def getPalmNorm(self, joint_gt: torch.Tensor,) -> torch.Tensor:
        palmNorm = unit_vector(torch.cross(joint_gt[:, 4] - joint_gt[:, 0], joint_gt[:, 7] - joint_gt[:, 4], dim=1))
        return palmNorm

    def restrainAbductionAngle(self,joints: torch.Tensor,bonelen: torch.Tensor,)->torch.Tensor:
        N = joints.shape[0]
        normidx = [1, 2, 2, 3]  # index,middle,ringy,pinky,thumb
        mcpidx = [1, 4, 10, 7]
        pipidx = [2, 5, 11, 8]
        loss,disEud=0,0
        r=6
        angleP = torch.tensor([np.pi/r+0.1890,np.pi/r+0.1331,np.pi/r-0.1491,np.pi/r+0.0347], device=joints.device, dtype=joints.dtype)
        angleN = torch.tensor([np.pi/r-0.1890,np.pi/r-0.1331,np.pi/r+0.1491,np.pi/r-0.0347], device=joints.device, dtype=joints.dtype)
        for i in range(4):
            palmNorm = self.getPalmNormByIndex(joints, normidx[i]).reshape(N, 3)  # palm up
            vh = palmNorm.reshape(N, 3)
            mcp = joints[:, mcpidx[i]].reshape(N,3)
            vd = -torch.sum(mcp * vh, dim=1).reshape(N, 1)
            pip = joints[:, pipidx[i]].reshape(N,3)
            wrist=joints[:,0]
            projpip=projectPoint2Plane(pip,vh,vd)[1].reshape(N,3)
            dis=euDist(mcp,pip).reshape(N)
            flexRatio=euDist(projpip,mcp).reshape(N)/(dis+epsilon)
            flexRatio[flexRatio<0.1]=0
            #valid=flexRatio>0.1
            #remove influence of perpendicular fingers
            #if(torch.sum(valid)):
            a=unit_vector(mcp-wrist).reshape(N,3)
            b=unit_vector(projpip-mcp).reshape(N,3)
            sign=torch.sum(torch.cross(a,b,dim=1)*palmNorm,dim=1)
            maskP=(sign>=0)
            maskN=(sign<0)
            angle = torch.acos(torch.clamp(torch.sum(a * b, dim=1), -1 + epsilon, 1 - epsilon)).reshape(-1)
            maskOver90=angle > 3.14 / 2
            angle[maskOver90] = 3.14 - angle[maskOver90]
            if(torch.sum(maskP)):
                cur=torch.max(angle[maskP] - angleP[i],
                          torch.zeros_like(angle[maskP])) * dis[maskP] * flexRatio[maskP]
                loss += torch.sum(cur)/N
                disEud+=torch.sum(cur*bonelen[maskP])/N
            if(torch.sum(maskN)):
                cur=torch.max(angle[maskN] - angleN[i],
                          torch.zeros_like(angle[maskN])) * dis[maskN] * flexRatio[maskN]
                loss += torch.sum(cur)/N
                disEud += torch.sum(cur * bonelen[maskN])/N
                #print("angle,idx,loss",angle,angle/3.14*180,mcpidx[i],curloss)
                #print('flexRatio,sign,maskOver90',flexRatio,sign,maskOver90)
        return loss/4,disEud/4 #constraint for 4 fingers

    def restrainFlexAngle(self, joints: torch.Tensor,bonelen: torch.Tensor,)->torch.Tensor:
        N = joints.shape[0]
        normidx=[-1,-1,-1,-1,0] #index,middle,ringy,pinky,thumb
        mcpidx=[1,4,10,7,13]
        stdFingerNorms=[]
        for i in range(5):
            palmNorm = self.getPalmNormByIndex(joints,normidx[i]).reshape(N, 3) # palm up
            vecWristMcp = unit_vector(joints[:, mcpidx[i]] - joints[:, 0]).reshape(N, 3) # wirst to mmcp
            stdFingerNorm = unit_vector(torch.cross(vecWristMcp,palmNorm, dim=1)) #direction from pmcp to imcp
            stdFingerNorms.append(stdFingerNorm.clone())
        jidx = [[0, 1, 2, 3, 17], [0, 4, 5, 6, 18], [0, 10, 11, 12, 19], [0, 7, 8, 9, 20],[0, 13,14,15,16]]
        fidces = [[0, 1, 2], [0, 1, 2],[0, 1, 2],[0, 1, 2],[1, 2],]
        loss,disEud = 0,0
        angleP = torch.tensor([np.pi/2, np.pi * 3 / 4,np.pi/2], device=joints.device, dtype=joints.dtype)
        angleN = torch.tensor([3.14 / 4, 3.14 / 18, 3.14 / 4], device=joints.device, dtype=joints.dtype)
        #angleN = torch.tensor([3.14 / 18, 3.14 / 18, 3.14 / 18], device=joints.device, dtype=joints.dtype)
        #csc todo: for test
        #angleN = torch.tensor([3.14 / 2, 3.14 / 4, 3.14 / 4], device=joints.device, dtype=joints.dtype)
        angleNthumb = torch.tensor([3.14 / 2, 3.14 / 4, 3.14 / 4], device=joints.device, dtype=joints.dtype)
        for fidx,finger in enumerate(jidx):
            if(fidx==4):angleN=angleNthumb
            for i in fidces[fidx]:
                a0, a1, a2 = joints[:, finger[i]], joints[:, finger[i + 1]], joints[:, finger[i + 2]].reshape(N, 3)

                a, b = unit_vector(a1 - a0), unit_vector(a2 - a1)
                removed = (torch.abs(torch.sum(a * b, dim=1)) > self.planeFlexRelax)
                disb=euDist(a1,a2).reshape(N)
                fingernorm = unit_vector(torch.cross(a, b, dim=1))

                sign = torch.sum(fingernorm * stdFingerNorms[fidx], dim=1).reshape(N)
                #print(fidx,torch.sum(a * b, dim=1))
                angle = torch.acos(torch.clamp(torch.sum(a * b, dim=1), -1 + epsilon, 1 - epsilon)).reshape(N)

                #print(finger[i:i+3],angle,angle/3.14*180,sign,disb)
                assert torch.sum(angle<0)==0
                angle[removed]=0
                #print("sign",sign)
                maskpositive=(sign>=0)
                masknegative=(sign<0)
                if(torch.sum(maskpositive)):
                    cur=torch.max(angle[maskpositive]-angleP[i],
                                                    torch.zeros_like(angle[maskpositive]))*disb[maskpositive]
                    loss+=torch.sum(cur)/N
                    disEud+=torch.sum(cur*bonelen[maskpositive])/N
                    #print(finger,i,torch.mean(cur*bonelen[maskpositive])*1000)
                    #cur0=torch.max(angle[maskpositive]-angleP[i],torch.zeros_like(angle[maskpositive]))
                    #cur1=disb[maskpositive]
                    #print(torch.mean(bonelen[maskpositive])*cur1*1000,cur0,angle,"pos")
                if(torch.sum(masknegative)):
                    #print(torch.mean(torch.max(angle[masknegative]-angleN[i],
                    #                              torch.zeros_like(angle[masknegative]))*disb[masknegative]))
                    cur=torch.max(angle[masknegative]-angleN[i],
                                                    torch.zeros_like(angle[masknegative]))*disb[masknegative]

                    loss += torch.sum(cur)/N
                    disEud+=torch.sum(cur*bonelen[masknegative])/N
                    #print(finger, i, torch.mean(cur * bonelen[masknegative])*1000)
                    # cur0 = torch.max(angle[masknegative] - angleP[i], torch.zeros_like(angle[masknegative]))
                    # cur1 = disb[masknegative]
                    #print(torch.mean(bonelen[masknegative])*cur1*1000,cur0,angle,"neg")
                #print('disEud',disEud*1000)
        return loss/15,disEud/15#15 joints constraints



if __name__ == "__main__":
    from cscPy.mano.network.manolayer import MANO_SMPL
    from cscPy.mano.network.utils import *
    import trimesh
    biolayer=BiomechanicalLayer(fingerFlexLoss=True,fingerAbductionLoss=True)

    mano_right = MANO_SMPL('/home/csc/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl', ncomps=45, oriorder=True,
                           device='cpu')
    rootr=torch.tensor(np.random.uniform(-0,0,[3]).astype(np.float32))
    pose=torch.tensor([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                       [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                       [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],dtype=torch.float32)

    # pose[0,2]-=1.57
    # pose[1,2]+=1.57/2
    # pose[2,2]-=1.57/2
    # pose[3,2]-=1.57
    # pose[4,2]-=1.57/2
    # for i in range(12):
    #     pose[i,2]+=1.57/1.5
    # pose[0, 1]+=np.pi/2/3
    # pose[3, 1]+=np.pi/2/3
    # pose[6, 1]+=np.pi/2/3
    # pose[9, 1]+=np.pi/2/3
    vertex_gt, joint_gt = \
                mano_right.get_mano_vertices(rootr.view(1, 1, 3),
                                             pose.view(1, 45),
                                             torch.zeros([10]).view(1, 10),
                                             torch.ones([1]).view(1, 1), torch.tensor([[0, 0, 0]]).view(1, 3),
                                             pose_type='euler', mmcp_center=False)

    print(biolayer(joint_gt))

    v = trimesh.Trimesh(vertices=vertex_gt[0].cpu().numpy(),faces=mano_right.faces)
    scene = trimesh.Scene(v)
    scene.show()


