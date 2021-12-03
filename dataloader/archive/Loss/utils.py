import torch
import numpy as np
from cscPy.Const.const import *
from progress.bar import Bar
from termcolor import colored, cprint

from torch.utils.tensorboard import SummaryWriter



def meanEuclideanLoss(pred, gt, scale, jointNr=21):
    pred = pred.view([-1, jointNr, 3])
    gt = gt.reshape([-1, jointNr, 3])
    eucDistance = torch.squeeze(torch.sqrt(1e-8+torch.sum((pred - gt) ** 2, dim=2)))
    meanEucDistance_normed = torch.mean(eucDistance)
    #print('scale',scale.shape,scale)
    eucDistance = eucDistance * torch.squeeze(scale).view(scale.shape[0], 1)
    meanEucDistance = torch.mean(eucDistance)
    return meanEucDistance_normed, meanEucDistance



def cloud_dis(c0, c1,scale):
    from chamfer_distance import ChamferDistance
    chamfer_dist = ChamferDistance()
    dist1, dist2 = chamfer_dist(c0, c1)
    dis0 = torch.mean(dist1, dim=1)
    dis1 = torch.mean(dist2, dim=1)
    cdlossEud = torch.mean(dis0 * scale.reshape(-1, 1)) + torch.mean(dis1 * scale.reshape(-1, 1))
    cdloss = torch.mean(dis0) + torch.mean(dis1)
    return cdloss,cdlossEud

def cloud_dis2(c0, c1,scale):
    N=c0.shape[0]
    c0=c0.reshape(N,906,1,3)
    c1=c1.reshape(N,1,906,3)
    dis=torch.sqrt(1e-8+torch.sum((c0-c1)**2,dim=3)).reshape(N,906,906)
    #print(torch.min(dis,dim=1))
    dis0=torch.min(dis,dim=1)[0]
    dis1=torch.min(dis,dim=2)[0]
    cdlossEud=torch.mean(dis0*scale.reshape(N,1))+torch.mean(dis1*scale.reshape(N,1))
    cdloss=torch.mean(dis0)+torch.mean(dis1)
    return cdloss,cdlossEud


def pose3d_loss(p0, p1, scale,jointN=21):
    pose_loss_rgb = torch.sum((p0 - p1) ** 2, dim=2)
    _, eucLoss_rgb = meanEuclideanLoss(p0, p1, scale,jointN)
    return torch.mean(pose_loss_rgb), eucLoss_rgb

def pose2d_loss(p0,p1):
    pose_loss = torch.sum((p0 - p1) ** 2, dim=2)
    eucLoss=torch.sqrt(pose_loss+epsilon)
    return torch.mean(pose_loss),torch.mean(eucLoss)

def getLatentLoss(z_mean, z_stddev, goalStd=1.0, eps=1e-9):
    latent_loss = 0.5 * torch.sum(z_mean**2 + z_stddev**2 - torch.log(z_stddev**2)  - goalStd, 1)
    return latent_loss


class LossHelper():
    def __init__(self,precision=3,useBar=True,usetb=False,summary="",dir="",logdir=''):
        self.loss={}
        self.precision=int(precision)
        self.useBar=useBar
        self.usetb=usetb
        if(usetb):
            print("use tensorboard")
            if(dir!="" or logdir!=""):
                import datetime
                self.tb = SummaryWriter(comment=summary, log_dir=os.path.join(logdir, dir, str(
                    datetime.datetime.now().date()) + summary))
            else:
                self.tb = SummaryWriter(comment=summary)
        self.bestvals={}

    def initForEachEpoch(self,lenFordataloader,summaryVariable=['epe','trepe','teepe','trpx2d','trZ','tepx2d','teZ'],
                         bestVal=['epe','trepe','teepe','trpx2d','trZ','tepx2d','teZ'],
                         median=['epe','trepe','teepe']):
        if(self.useBar):
            self.bar = Bar(colored('T', 'yellow'), max=lenFordataloader)
        self.summaryVariable=summaryVariable
        self.bestVal=bestVal
        self.median=median
        #for name in self.bestVal:self.bestvals[name]=100000.0
    def add(self,dic):
        for name,value in dic.items():
            if (name == 'epoch' or name.startswith('iter')):v=int(value)
            else:v=float(value)
            if(name in self.loss):
                self.loss[name].append(v)
            else:
                self.loss[name]=[v] 
    def get(self,name='epe'):
        for iname in self.loss:
            if(iname==name):return np.mean(self.loss[iname])
        return -1
    def show(self):
        for name in self.loss:
            if(name in self.bestVal):
                ave=float(np.mean(self.loss[name]))
                if(name not in self.bestvals):
                    self.bestvals[name] = ave
                else:
                    self.bestvals[name]=min(self.bestvals[name],ave)
                print(name,'best mean',self.bestvals[name])
            if (self.usetb and name in self.summaryVariable):
                print(name, 'mean_', np.mean(self.loss[name]))
                self.tb.add_scalar(name, np.mean(self.loss[name]), self.loss['epoch'][-1])
                if(name in self.median):
                    #self.tb.add_scalar(name, np.median(self.loss[name]), self.loss['epoch'][-1])
                    print(name, 'median', np.median(self.loss[name]))
            elif(not (name == 'epoch' or name.startswith('iter'))):
                print(name,'notb:mean',np.mean(self.loss[name]))
    def showcurrent(self):
        if(self.useBar):
            self.bar.suffix = (str(self))
            self.bar.next()
        else:
            print(str(self))
    def getinfo(self):
        out = ""
        cnt = 0
        for name in self.loss:
            if (name == 'epoch' or name.startswith('iter')):
                out += name+":"+str(self.loss[name][-1])+" "
            else:
                txt = '{0:.' + str(self.precision) + 'f}'
                out += name + " " + txt.format(np.mean(self.loss[name])) + " "
            cnt += 1
            if (cnt >= 10):
                if(self.useBar):out+=' '
                else:out += '\n'
                cnt = 0
        return out
    def __str__ (self):
        return self.getinfo()
    def finish(self):
        self.show()
        if self.useBar:self.bar.finish()
        self.loss={}