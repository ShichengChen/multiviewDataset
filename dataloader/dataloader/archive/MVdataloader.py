from torch.utils.data import Dataset
from cscPy.multiviewDataset.toolkit import MultiviewDatasetDemo
import numpy as np
from cscPy.handProcess.dataAugment import processing_augmentation
from cscPy.handProcess.preprocessImages import preprocessMVdataset
import torchvision
import cv2
from cscPy.mano.network.utils import *
from cscPy.mano.network.utilsSmallFunctions import *
from cscPy.mano.network.Const import *
from cscPy.globalCamera.camera import CameraIntrinsics,perspective_projection



class MVDataloader(Dataset):
    def __init__(self,file_path=None,train=True):
        self.train=train
        self.demo=MultiviewDatasetDemo(loadMode=False)
        self.num_samples=self.demo.N*4


    def __len__(self):
        return self.num_samples#//400

    def __getitem__(self, idx):
        vi=idx%4
        id=idx//4
        cloud=self.demo.get4viewCloud(id,sampleN=906,disprio=1.5)[vi][:,:3,0]
        depth=255-self.demo.getBetterDepth(id,uselist=True)[vi]
        pose_uv=self.demo.getPose2D(id,vi)
        joint_gt=self.demo.getPose3D(id,vi)
        image = depth.reshape(480, 640, 3)
        # image=self.demo.getImgsList(id)[vi]

        joint_gt = joint_gt[MV2mano_skeidx]
        pose_uv = pose_uv[MV2mano_skeidx]


        # pose_uv = perspectiveProjection(joint_gt.copy(), self.demo.Ks[vi])[:, :2].astype(int)
        # for i in range(pose_uv.shape[0]):
        #     image=cv2.circle(image, (pose_uv[i, 0], pose_uv[i, 1]), 3, (255,0,0))
        #     # image=cv2.putText(image,str(i),(pose_uv[i, 0], pose_uv[i, 1]),cv2.FONT_HERSHEY_SIMPLEX,
        #     #                   1,(255))
        # cv2.imshow('img', image)
        # cv2.waitKey(0)

        # joint_gt and cloud from mm to m
        joint_gt = joint_gt / 1000
        cloud = cloud / 1000


        image_crop, pose3d_normed,cloud_norm, scale, root = preprocessMVdataset(image, pose_uv, joint_gt,cloud, cidx=4, cnxt=5)



        # image_crop, depth, cloud, pose3d = \
        #     processing_augmentation(image_crop, None,vertex_gt,joint_gt,self.train)
        image_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        image_crop = image_trans(torchvision.transforms.ToPILImage()((image_crop).astype(np.uint8)))

        # cur = image_crop.permute(1, 2, 0).numpy()
        # cur = (cur * 255).astype(np.uint8)
        # cv2.imshow('cur', cur)
        # cv2.waitKey(0)

        out = {"img": get32fTensor(image_crop.reshape(3, 256, 256)), "pose3d": get32fTensor(pose3d_normed).reshape(21,3),
               "scale": get32fTensor(scale).reshape(1,1), "root": get32fTensor(root).reshape(1,3),
               "cloud": get32fTensor(cloud_norm).reshape(906,3),
               "mask": torch.tensor([0]).long(),'idx':torch.tensor([idx]).long(),
               'K':get32fTensor(self.demo.Ks[vi]).reshape(1,3,3),'pose2d':get32fTensor(pose_uv)}
        return out


if __name__ == "__main__":
    mvdataloader=MVDataloader()
    #hole=RandomHoles(640,480)
    for i in range(20):
        dms=mvdataloader.__getitem__(i)


