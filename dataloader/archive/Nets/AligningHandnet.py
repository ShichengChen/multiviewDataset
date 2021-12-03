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

class Point_conv(nn.Module):
    def __init__(self,ip,op,act='relu',bn=True):
        super(Point_conv, self).__init__()
        l = [nn.Conv2d(ip, op, kernel_size=1, padding=0)]
        #if bn: l.append(nn.BatchNorm2d(op, eps=1e-3, momentum=1e-2))
        if bn: l.append(nn.BatchNorm2d(op))
        if (act == 'relu'):
            l.append(nn.ReLU())
        elif (act == 'sigmoid'):
            l.append(nn.Sigmoid())
        elif (act == 'tanh'):
            l.append(nn.Tanh())
        elif (act == 'no'):
            pass
        else:
            assert (False), 'no such act'
        self.f=nn.Sequential(*l)

    def forward(self,x):
        #print('self.f(x)[0,0,0]',self.f(x)[0,0,0])
        return self.f(x)

class EltwiseLayer(nn.Module):
    def __init__(self,op):
        super(EltwiseLayer, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(torch.ones([1,op,1,1],requires_grad=True)))
    def forward(self, x):
        return x * self.weights

class Pointnet_residual_global(nn.Module):
    def __init__(self,ip,op):
        super(Pointnet_residual_global, self).__init__()
        self.op = op
        self.p=nn.ModuleList([nn.MaxPool1d(POINT_SIZE),nn.MaxPool1d(POINT_SIZE),
                              nn.MaxPool1d(POINT_SIZE),nn.MaxPool1d(POINT_SIZE)])
        self.a=nn.ModuleList([EltwiseLayer(op),EltwiseLayer(op),EltwiseLayer(op),EltwiseLayer(op)])
        self.b=nn.ModuleList([EltwiseLayer(op),EltwiseLayer(op),EltwiseLayer(op),EltwiseLayer(op)])
        self.pconv=nn.ModuleList([Point_conv(ip,op),Point_conv(ip,op),Point_conv(ip,op),Point_conv(ip,op)])

    def forward(self,x):
        input=x
        for i in range(4):
            maxp=self.p[i](x.view(-1,self.op,POINT_SIZE)).view(-1,self.op,1,1)
            x=self.a[i](x)-self.b[i](maxp)
            x=self.pconv[i](x)
        return x+input


class encCouldNet(nn.Module):
    def __init__(self, n_latent=64):
        super(encCouldNet, self).__init__()
        channel=[3,64,256,512]
        c2=[512,256,128,128]
        a=['','relu','relu','sigmoid']
        a2=['','relu','relu','no']
        bn=['',True,True,False]
        l,l1,l2=[],[],[]
        for i in range(1,len(channel)):
            l.append(Point_conv(channel[i-1],channel[i]))
            for j in range(2):
                l.append(Pointnet_residual_global(channel[i], channel[i]))
        for i in range(1,len(c2)):
            l1.append(Point_conv(c2[i-1],c2[i],act=a[i],bn=bn[i]))
            l2.append(Point_conv(c2[i - 1], c2[i],act=a2[i],bn=bn[i]))
        self.f=nn.Sequential(*l)
        self.s=nn.Sequential(*l1)
        self.v=nn.Sequential(*l2)

        self.bnl=nn.Sequential(nn.BatchNorm1d(128),nn.Linear(128,128),nn.ReLU())
        self.l3=nn.Sequential(nn.Linear(128,n_latent),nn.Tanh())
        self.l4=nn.Sequential(nn.Linear(128,n_latent),nn.Softplus())


    def forward(self, x, training):
        x = x[:, :,:POINT_SIZE]
        x = x.view([-1,3,1,POINT_SIZE])
        x=self.f(x)
        #print('x0', x.shape)
        score=self.s(x)+1e-7
        value=self.v(x)
        scoresMax,_=torch.max(score,dim=3)
        scoresMax=scoresMax.view(-1,128,1,1)
        score=score/(scoresMax)
        weightSum = torch.squeeze(torch.sum(score, axis=3))
        x=torch.squeeze(torch.sum(score*value,dim=3))/(weightSum)
        #print('x',x.shape)
        x = x.view([-1, 128])
        x=self.bnl(x)
        #print('x', x.shape)
        mn = 2.0 * self.l3(x)
        sd = 1e-9 + self.l4(x)
        sd = torch.clamp(sd, 1e-9, 100)

        m = torch.distributions.normal.Normal(torch.zeros_like(sd), torch.ones_like(sd))
        epsilon = m.sample()

        if training:
            z=mn+sd*epsilon
        else:
            z=mn
        return z,mn,sd


class decCloudNet(nn.Module):
    def __init__(self, n_latent=64):
        super(decCloudNet, self).__init__()
        self.n_latent=n_latent
        c1=[n_latent+2,256,512,1024,512,3]
        a=['relu','relu','relu','relu','tanh']
        c2=[n_latent+3,256,512,1024,512,3]
        l1,l2=[],[]
        for i in range(1,len(c1)):
            l1.append(Point_conv(c1[i-1],c1[i],act=a[i-1]))
        for i in range(1,len(c2)):
            if(i!=len(c2)-1):
                l2.append(Point_conv(c2[i-1],c2[i],act=a[i-1]))
            else:
                l2.append(Point_conv(c2[i - 1], c2[i], act=a[i - 1],bn=False))
        self.l1=nn.Sequential(*l1)
        self.l2=nn.Sequential(*l2)

        u_ = np.transpose(np.tile(np.arange(-DecUVSize / 2 + 0.5, DecUVSize / 2, 1, dtype=float), np.int(DecUVSize)))
        v_ = np.transpose(np.repeat(np.arange(-DecUVSize / 2 + 0.5, DecUVSize / 2, 1, dtype=float), np.int(DecUVSize)))
        u = np.array(u_ / (DecUVSize * 0.5), dtype=np.float32) * 10.0
        v = np.array(v_ / (DecUVSize * 0.5), dtype=np.float32) * 10.0
        self.uv=np.stack([u, v]).reshape([-1, 2,DecPointSize])
        self.quan = Quantization_module()

    def forward(self, x):
        x=self.quan(x)
        coordinate = self.uv.copy().repeat(x.shape[0],axis=0)
        #coordinate=(torch.rand([x.shape[0],2,DecPointSize],dtype=torch.float32,device=x.device)-0.5)*2
        coordinate=torch.tensor(coordinate,device=x.device)
        x = x.view(x.shape[0], self.n_latent,1)
        x=x.repeat(1, 1, DecPointSize)
        y = torch.cat([x, coordinate], dim=1).view([-1, self.n_latent + 2,DecPointSize, 1])
        y=self.l1(y)
        y=y.view(x.shape[0],3,DecPointSize)
        y=torch.cat([x,y],dim=1).reshape(-1, self.n_latent+3,DecPointSize, 1)
        y=self.l2(y)
        cloud = y.view(x.shape[0], 3,DecPointSize)
        return cloud


class encoderRGB(nn.Module):
    def __init__(self, n_latent=64):
        super(encoderRGB, self).__init__()
        self.res=models.resnet18(pretrained=True)
        self.res.fc=nn.Sequential(nn.Linear(512,1000))
        self.l=nn.Sequential(nn.BatchNorm1d(1000),nn.ReLU(),nn.Linear(1000,128))
        self.l1 = nn.Sequential(nn.Linear(128, n_latent), nn.Tanh())
        self.l2 = nn.Sequential(nn.Linear(128, n_latent), nn.Softplus())
        #self.l=nn.Sequential(nn.BatchNorm1d(1000),nn.ReLU(),nn.Linear(1000,n_latent))
    def forward(self, x,training):
        x=self.res(x)
        x=self.l(x)
        mn = self.l1(x) * 2.0
        sd = torch.clamp(self.l2(x) + 1e-9, 1e-9, 100)
        m = torch.distributions.normal.Normal(torch.zeros_like(mn), torch.ones_like(mn))
        epsilon = m.sample()
        if training:
            z = mn + sd * epsilon
        else:
            z = mn
        return z,mn,sd

class decPoseNet(nn.Module):
    def __init__(self, n_latent=64,jointN=21):
        super(decPoseNet, self).__init__()
        self.l=nn.Sequential(nn.Linear(n_latent,128),nn.ReLU(),
                             nn.Linear(128,128),nn.ReLU(),
                             nn.Linear(128,128),nn.ReLU(),
                             nn.Linear(128,jointN*3),)
    def forward(self, x):
        return self.l(x)

def gaussianProd(mn_1, sd_1, mn_2, sd_2, training, eps=1e-8):
    T0 = torch.ones_like(sd_1)
    T1 = 1. / (sd_1 + eps)
    T2 = 1. / (sd_2 + eps)
    mn_prod = (mn_1 * T1 + mn_2 * T2) / (T0 + T1 + T2)
    sd_prod = 1. / (T0 + T1 + T2)

    m = torch.distributions.normal.Normal(torch.zeros_like(mn_prod), torch.ones_like(mn_prod))
    epsilon = m.sample()
    if training:
        z_prod= mn_prod + sd_prod * epsilon
    else:
        z_prod = mn_prod
    return z_prod, mn_prod, sd_prod




_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

class decHeatmapNet(nn.Module):
    def __init__(self, n_latent=64, batch_size=32, crop_size=64, channel_size=21, name='dec_heatmap'):
        super(decHeatmapNet, self).__init__()
        self.name = name
        self._CROP_SIZE = crop_size
        self.gf_dim = n_latent
        self.channel_size = channel_size
        self.batch_size = batch_size
        s_h, s_w = self._CROP_SIZE, self._CROP_SIZE
        def conv_out_size_same(size, stride):
            return int(math.ceil(float(size) / float(stride)))
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        self.s_h8, self.s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        self.l0=nn.Linear(n_latent,self.gf_dim * 4 * self.s_h8 * self.s_w8)
        self.l1=nn.Sequential(nn.ReLU(),nn.BatchNorm2d(self.gf_dim*4),
                              nn.ConvTranspose2d(self.gf_dim*4,self.gf_dim*2,5,2,4),
                              nn.ReLU(),nn.BatchNorm2d(self.gf_dim*2),
                              nn.ConvTranspose2d(self.gf_dim*2,self.gf_dim,5,2,4,1),
                              nn.ReLU(),nn.BatchNorm2d(self.gf_dim),
                              nn.ConvTranspose2d(self.gf_dim,self.channel_size,5,2,4,1),
                              nn.Sigmoid())

    def forward(self, x):
        x=self.l0(x).view([-1, self.s_h8, self.s_w8, self.gf_dim * 4]).permute(0,3,1,2)
        x=self.l1(x)
        return x






##############################################################33
# this implementation is from
#https://github.com/YanWei123/Pytorch-implementation-of-FoldingNet-encoder-and-decoder-with-graph-pooling-covariance-add-quanti




class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x
class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # x = batch,1024,n(n=2048)
        x = torch.max(x, 2, keepdim=True)[0]  # x = batch,1024,1
        x = x.view(-1, 1024)  # x = batch,1024
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans

def GridSamplingLayer(batch_size, meshgrid):
    ret = np.meshgrid(*[np.linspace(it[0], it[1], num=it[2]) for it in meshgrid])
    ndim = len(meshgrid)
    grid = np.zeros((np.prod([it[2] for it in meshgrid]), ndim), dtype=np.float32)  # MxD
    for d in range(ndim):
        grid[:, d] = np.reshape(ret[d], -1)
    g = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)
    return g

class FoldingNetDecFold1(nn.Module):
    def __init__(self):
        super(FoldingNetDecFold1, self).__init__()
        self.conv1 = nn.Conv1d(514, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)
        self.relu = nn.ReLU()
    def forward(self, x):  # input x = batch,514,45^2
        x = self.relu(self.conv1(x))  # x = batch,512,45^2
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
class FoldingNetDecFold2(nn.Module):
    def __init__(self):
        super(FoldingNetDecFold2, self).__init__()
        self.conv1 = nn.Conv1d(515, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)
        self.relu = nn.ReLU()
    def forward(self, x):  # input x = batch,515,45^2
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class FoldingNetEnc(nn.Module):
    def __init__(self):
        super(FoldingNetEnc, self).__init__()
        self.feat = PointNetfeat(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
    def forward(self, x):
        x, trans = self.feat(x)  # x = batch,1024
        x = F.relu(self.bn1(self.fc1(x)))  # x = batch,512
        x = self.fc2(x)  # x = batch,512
        return x
from torch.autograd import Function
class Quantization(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.round(input)
        return output
    @staticmethod
    def backward(ctx,grad_output):
        return grad_output
class Quantization_module(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return Quantization.apply(input)

class FoldingNetDec(nn.Module):
    def __init__(self,n_latent=64):
        super(FoldingNetDec, self).__init__()
        self.fold1 = FoldingNetDecFold1()
        self.fold2 = FoldingNetDecFold2()
        self.quan = Quantization_module()
        self.l=nn.Linear(n_latent,512)
    def forward(self, x):  # input x = batch, 512
        x=self.l(x)
        x=self.quan(x)
        batch_size = x.size(0)
        x = torch.unsqueeze(x, 1)  # x = batch,1,512
        x = x.repeat(1, 256, 1)  # x = batch,45^2,512
        code = x
        code = x.transpose(2, 1)  # x = batch,512,45^2
        meshgrid = [[-0.3, 0.3, 16], [-0.3, 0.3, 16]]
        grid = GridSamplingLayer(batch_size, meshgrid)  # grid = batch,45^2,2
        grid = torch.from_numpy(grid)
        if x.is_cuda:
            grid = grid.cuda()
        x = torch.cat((x, grid), 2)  # x = batch,45^2,514
        x = x.transpose(2, 1)  # x = batch,514,45^2
        x = self.fold1(x)  # x = batch,3,45^2
        p1 = x  # to observe
        x = torch.cat((code, x), 1)  # x = batch,515,45^2
        x = self.fold2(x)  # x = batch,3,45^2
        return x
