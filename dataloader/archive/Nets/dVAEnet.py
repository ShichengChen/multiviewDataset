import torch
from torch import nn
from torch.autograd import Variable
import torch.distributions.normal
import numpy as np
import math
import torchvision.models as models
import torch.nn.functional as F
POINT_SIZE = 256
DecPointSize = 256
DecUVSize = np.sqrt(DecPointSize)
import torch
from torch import nn
from torch.autograd import Variable
import torch.distributions.normal
import numpy as np
import math

class SixLinear(nn.Module):
    def __init__(self,id,d=128):
        super(SixLinear,self).__init__()
        self.l = nn.Sequential(nn.Linear(id, d),nn.BatchNorm1d(d), nn.ReLU(),
                               nn.Linear(d, d),nn.BatchNorm1d(d), nn.ReLU(),
                               nn.Linear(d, d),nn.BatchNorm1d(d), nn.ReLU(),
                               nn.Linear(d, d),nn.BatchNorm1d(d), nn.ReLU(),
                               nn.Linear(d, d),nn.BatchNorm1d(d), nn.ReLU(),
                               nn.Linear(d, d))
    def forward(self, x):return self.l(x)

class decCPoseNet(nn.Module):
    def __init__(self, n_latent=16):
        super(decCPoseNet, self).__init__()
        self.l=nn.Sequential(SixLinear(n_latent), nn.Linear(128, 21*3),)
    def forward(self, x):
        return self.l(x).view(-1,21*3)
class decPoseNet(nn.Module):
    def __init__(self, n_latent=32):
        super(decPoseNet, self).__init__()
        self.l=nn.Sequential(SixLinear(n_latent), nn.Linear(128, 21*3),)
    def forward(self, x):
        return self.l(x).view(-1,21*3)
class decViewNet(nn.Module):
    def __init__(self, n_latent=16):
        super(decViewNet, self).__init__()
        self.l=nn.Sequential(SixLinear(n_latent), nn.Linear(128, 3*3),)
    def forward(self, x):
        return self.l(x).view(-1,3, 3)

class encCPoseNet(nn.Module):
    def __init__(self, id=21*3,n_latent=16):
        super(encCPoseNet, self).__init__()
        self.l = nn.Sequential(SixLinear(id),nn.BatchNorm1d(128), nn.Linear(128, 128), nn.ReLU())
        self.l1 = nn.Sequential(nn.Linear(128, n_latent), nn.Tanh())
        self.l2 = nn.Sequential(nn.Linear(128, n_latent), nn.Softplus())
    def forward(self, x,training):
        x = self.l(x.view(-1, 21*3))
        mn = self.l1(x) * 2.0
        sd = torch.clamp(self.l2(x) + 1e-9, 1e-9, 100)
        m = torch.distributions.normal.Normal(torch.zeros_like(mn), torch.ones_like(mn))
        epsilon = m.sample()
        if training:
            z = mn + sd * epsilon
        else:
            z = mn
        return z, mn, sd

class encViewNet(nn.Module):
    def __init__(self, id=3*3,n_latent=16):
        super(encViewNet, self).__init__()
        self.l = nn.Sequential(SixLinear(id),nn.BatchNorm1d(128), nn.Linear(128, 128), nn.ReLU())
        self.l1 = nn.Sequential(nn.Linear(128, n_latent), nn.Tanh())
        self.l2 = nn.Sequential(nn.Linear(128, n_latent), nn.Softplus())
    def forward(self, x,training):
        x=self.l(x.view(-1,9))
        mn = self.l1(x) * 2.0
        sd = torch.clamp(self.l2(x) + 1e-9, 1e-9, 100)
        m = torch.distributions.normal.Normal(torch.zeros_like(mn), torch.ones_like(mn))
        epsilon = m.sample()
        if training:
            z = mn + sd * epsilon
        else:
            z = mn
        return z, mn, sd

class encoderRGB(nn.Module):
    def __init__(self, n_latent=32):
        super(encoderRGB, self).__init__()
        self.res=models.resnet18(pretrained=True).cuda()
        self.res.fc=nn.Sequential(nn.Linear(512,1000))
        self.l = nn.Sequential(nn.BatchNorm1d(1000), nn.Linear(1000, 128))
        self.l1 = nn.Sequential(nn.Linear(128, n_latent), nn.Tanh())
        self.l2 = nn.Sequential(nn.Linear(128, n_latent), nn.Softplus())
    def forward(self, x,training):
        x = self.res(x)
        x = self.l(x)
        mn = self.l1(x) * 2.0
        sd = torch.clamp(self.l2(x) + 1e-9, 1e-9, 100)
        m = torch.distributions.normal.Normal(torch.zeros_like(mn), torch.ones_like(mn))
        epsilon = m.sample()
        if training:
            z = mn + sd * epsilon
        else:
            z = mn
        return z, mn, sd

import utils
class VAE(nn.Module):
    def __init__(self, n_latent=32):
        super(VAE, self).__init__()
        self.enCpose=encCPoseNet()
        self.enV=encViewNet()
        self.enRGB=encoderRGB()
        self.dPose=decPoseNet()
        self.dCpose=decCPoseNet()
        self.dV=decViewNet()


    def forward(self, img,cpose,mat, phase,training):
        if(phase==0):
            zp,mnp,sdp=self.enCpose(cpose,training)
            zv,mnv,sdv=self.enV(mat,training)

            z=torch.cat([zp,zv],dim=1)
            pose=self.dPose(z)
            cpose=self.dCpose(zp)
            view=self.dV(zv)

            return mnp, sdp,mnv,sdv, pose, cpose, view
        else:
            z,mn,sd=self.enRGB(img,training)
            pose=self.dPose(z)
            cpose=self.dCpose(z[:,:16])
            view=self.dV(z[:,16:])
            return mn,sd,pose,cpose,view

