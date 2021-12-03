import os
import numpy as np
import cv2
import pickle
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy import ndimage
import torchvision, scipy.misc, imageio
from PIL import Image
import math
import matplotlib.pyplot as plt
import pickle
'''
###############################^M
### GANerated Hands Dataset ###^M
###############################^M
frames without object: 143,449^M
frames with object: 188,050^M
^M
camera intrinsics:^M
[ 617.173,       0, 315.453,^M
        0, 617.173, 242.259,^M
        0,       0,       1]^M
^M
Directory structure:^M
* joints.png^M
* license.txt^M
* README.txt (this file)^M 
* data/^M
        * withObject / noObject^M
                * <partition>/ (s.t. every directory contains 1024 example frames)^M
^M
For every frame, the following data is provided:^M
- color:                        24-bit color image of the hand (cropped)^M
- joint_pos:            3D joint positions relative to the middle MCP joint. The values are normalized such that the length between middle finger MCP and wrist is 1.
                                        The positions are organized as a linear concatenation of the x,y,z-position of every joint (joint1_x, joint1_y, joint1_z, joint2_x, ...). ^M
                                        The order of the 21 joints is as follows: W, T0, T1, T2, T3, I0, I1, I2, I3, M0, M1, M2, M3, R0, R1, R2, R3, L0, L1, L2, L3.^M
                                        Please also see joints.png for a visual explanation.
- joint2D:                      2D joint positions in u,v image coordinates. The positions are organized as a linear concatenation of the u,v-position of every joint (joint1_u, joint1_v, joint2_u, …).^M
- joint_pos_global:     3D joint positions in the coordinate system of the original camera (before cropping)^M
- crop_params:          cropping parameters that were used to generate the color image (256 x 256 pixels) from the original image (640 x 480 pixels), ^M
                                        specified as top left corner of the bounding box (u,v) and a scaling factor
~                                                                                                                                                                                                           
~                                                                                                                                                                                                           
~                                                                                                                         
'''
class GHDDateset(Dataset):
    def __init__(self, path_name='/mnt/data/csc/GANeratedHands_Release/data/', train=True, type=0,image_size = 64,traintestsplit=0.8):
        print("load GHD")
        if (train):
            self.mode = "training"
        else:
            self.mode = "eval"
        self.path_name = path_name
        self.image_size = image_size
        typefolder=[['noObject'],['withObject'],['withObject','noObject']]
        self.anno_all=[]
        # for f in typefolder[type]:
        #     base=os.path.join(path_name,f)
        #     print(base)
        #     for (dirpath, dirnames, filenames) in os.walk(base):
        #         prefix = list(set([i[:4] for i in filenames]))
        #         for pre in prefix:
        #             cur={}
        #             prepath=os.path.join(dirpath,pre)
        #             #print(prepath)
        #             with open(prepath+'_joint_pos.txt') as f:
        #                 content = f.readlines()[0]
        #                 content = [float(x.strip()) for x in content.split(',')]
        #                 cur['relativeJoints']=np.array(content).reshape(21,3)
        #             with open(prepath+'_crop_params.txt') as f:
        #                 content = f.readlines()[0]
        #                 content = [float(x.strip()) for x in content.split(',')]
        #                 cur['scale']=np.array(content[-1])
        #                 cur['sxy']=np.array(content[:2])
        #             with open(prepath+'_joint2D.txt') as f:
        #                 content = f.readlines()[0]
        #                 content = [float(x.strip()) for x in content.split(',')]
        #                 cur['uv']=np.array(content).reshape(21,2)
        #             with open(prepath+'_joint_pos_global.txt') as f:
        #                 content = f.readlines()[0]
        #                 content = [float(x.strip()) for x in content.split(',')]
        #                 cur['xyz']=np.array(content).reshape(21,3)
        #             cur['path']=prepath+'_color_composed.png'
        #             self.anno_all.append(cur)
        # with open(path_name+'_'.join(typefolder[type])+'.pkl', 'wb') as f:
        #     print(path_name+'_'.join(typefolder[type])+'.pkl')
        #     pickle.dump(self.anno_all, f)
        savepath=path_name + '_'.join(typefolder[type]) + '.pkl'
        savepath=savepath+'_small.pkl'
        print(savepath)
        with open(savepath, 'rb') as f:
            self.anno_all=pickle.load(f)
        # with open(savepath+'_small.pkl', 'wb') as f:
        #     print(savepath+'_small.pkl')
        #     pickle.dump(self.anno_all[:int(globalVariables.numberofsamples/0.79)], f)

        alllen=len(self.anno_all)
        if(train):self.anno_all=self.anno_all[:int(alllen*traintestsplit)]
        else:self.anno_all=self.anno_all[int(alllen*traintestsplit):]
        self.num_samples=len(self.anno_all)

    def __len__(self):
        '''
        251752746 /mnt/data/csc/GANeratedHands_Release/data/noObject.pkl
        114759
        330403691 /mnt/data/csc/GANeratedHands_Release/data/withObject.pkl
        150440
        582156972 /mnt/data/csc/GANeratedHands_Release/data/withObject_noObject.pkl
        265199
        '''
        return self.num_samples

    def __getitem__(self, idx):
        print('GHD', idx)
        anno = self.anno_all[idx]
        image = cv2.imread(anno['path'])
        uv = anno['uv']
        uv_vis = anno['uv'][:, -1].astype(int) != 0
        pose3d = anno['xyz']
        print(anno.keys())
        print(anno['relativeJoints'])
        print(anno['scale'])
        print(anno['sxy'])

        ghd2rhd = np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17])
        uv=uv[ghd2rhd]
        pose3d=pose3d[ghd2rhd]

        camera = np.array([[617.173,0, 315.453],[0, 617.173, 242.259],[0,0,1]])
        hand_side=True
        test3Dmatch2D=False
        if test3Dmatch2D:
            uvd_point = np.zeros([uv.shape[0],3])
            uvd_point[:, 0] = pose3d[:, 0] * camera[0,0] / pose3d[:, 2] + camera[0,-1]
            uvd_point[:, 1] = pose3d[:, 1] * camera[1,1] / pose3d[:, 2] + camera[1,-1]
            uvd_point[:, 2] = pose3d[:, 2]
            uvd_point=uvd_point[:,:-1]
            #print(uvd_point,anno['scale'])
            uvd_point=(uvd_point-anno['sxy'])*anno['scale']
            #print(image)
            plt.figure()
            plt.subplot(1, 1, 1)
            plt.axis('off')
            plt.imshow(image)
            #plt.scatter(uv[:, 0], uv[:, 1], alpha=0.6)
            plt.scatter(uvd_point[:, 0], uvd_point[:, 1], alpha=0.9)
            plt.show()
        hand_image, new_uv, hand_side, uv_vis, relative_depth, pose3d_root, bone_length, pose3d = \
            self.segment_hand(image, uv, uv_vis, pose3d, hand_side=hand_side)
        image_trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])

        if self.mode == 'training':
            hand_image, new_uv = self.augmentation(hand_image, new_uv)
            image_trans = transforms.Compose(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                 transforms.RandomErasing(p=0.75, scale=(0.02, 0.1)),
                 ])

        hand_image = Image.fromarray(hand_image)
        hand_image = image_trans(hand_image).float()

        new_uv = torch.from_numpy(new_uv).float()
        uv_vis = torch.from_numpy(uv_vis).float()
        relative_depth = torch.from_numpy(relative_depth).float()
        pose3d = torch.from_numpy(pose3d).float()

        hand_side, pose3d_root, bone_length, camera = \
            torch.as_tensor(hand_side).bool(),torch.as_tensor(pose3d_root).float(),torch.as_tensor(bone_length).float(),\
            torch.from_numpy(camera).float()

        crop_center, crop_size={},{}
        #return image, hand_image, new_uv, relative_depth, uv_vis, hand_side, pose3d_root, bone_length, camera, pose3d
        return image, hand_image, new_uv, relative_depth, uv_vis, crop_center, crop_size, hand_side, pose3d_root, bone_length, camera, pose3d

    def augmentation(self, image, uv):
        height, width, _ = image.shape
        randScaleImage = np.random.uniform(low=0.8, high=1.2)
        randAngle = 2 * math.pi * np.random.rand(1)
        randTransX = np.random.uniform(low=-5., high=5.)
        randTransY = np.random.uniform(low=-5., high=5.)

        rot_mat = cv2.getRotationMatrix2D((width/2, height/2), -180.0 * randAngle[0]/ math.pi, randScaleImage)
        rot_mat[0,2] += randTransX
        rot_mat[1,2] += randTransY

        image_aug = image.copy()
        uv_aug = np.ones([uv.shape[0], uv.shape[1] + 1])
        uv_aug[:, :2] = uv

        image_aug = cv2.warpAffine(image_aug, rot_mat, (width, height), flags=cv2.INTER_NEAREST, borderValue=0.0)
        uv_aug = np.dot(rot_mat, uv_aug.T).T

        return image_aug, uv_aug

    def segment_hand(self, image, kp_coord_uv, kp_visible, pose3d, hand_side):
        if hand_side:
            pose_uv_all = kp_coord_uv[:21, :]
            uv_vis = kp_visible[:21]
            pose3d = pose3d[:21]
        else:
            pose_uv_all = kp_coord_uv[-21:, :]
            uv_vis = kp_visible[-21:]
            pose3d = pose3d[-21:]

        pose_uv_vis = pose_uv_all[uv_vis, :]
        # resize image to the input size of hourglass
        height, width, _ = image.shape
        refine_size = self.image_size
        image_crop = cv2.resize(image, (refine_size, refine_size), interpolation=cv2.INTER_NEAREST)

        crop_scale = refine_size/256
        new_uv = self.creat_refine_uv(pose_uv_all, crop_scale, uv_vis)

        if hand_side:  # transfer hands to right hand
            image_crop = cv2.flip(image_crop, 1)
            new_uv[:, 0] = refine_size - new_uv[:, 0]

        pose3d_root = pose3d[9, :]  # this is the root coord
        pose3d_rel = pose3d - pose3d_root  # relative coords in metric coords
        bone_length = np.sqrt(np.sum(np.square(pose3d_rel[9, :] - pose3d_rel[0, :])))
        pose3d_normed = pose3d_rel / bone_length
        relative_depth = pose3d_normed[:, 2]

        return image_crop, new_uv, hand_side, uv_vis, relative_depth, pose3d_root, bone_length, pose3d

    def imcrop(self, img, center, crop_size):
        x1 = int(center[0] - crop_size)
        y1 = int(center[1] - crop_size)
        x2 = int(center[0] + crop_size)
        y2 = int(center[1] + crop_size)

        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            img, x1, x2, y1, y2 = self.pad_img_to_fit_bbox(img, x1, x2, y1, y2)

        if img.ndim < 3:
            img_crop = img[y1:y2, x1:x2]
        else:
            img_crop = img[y1:y2, x1:x2, :]

        return img_crop

    def pad_img_to_fit_bbox(self, img, x1, x2, y1, y2):
        if img.ndim < 3:  # for depth
            borderValue = [0]
        else:  # for rgb
            borderValue = [127, 127, 127]

        img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                                 -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_CONSTANT, value=borderValue)
        y2 += -min(0, y1)
        y1 += -min(0, y1)
        x2 += -min(0, x1)
        x1 += -min(0, x1)
        return img, x1, x2, y1, y2

    def creat_refine_uv(self, kp_coord_uv, crop_scale, kp_visible):
        keypoint_uv21_u = (kp_coord_uv[:, 0]) * crop_scale
        keypoint_uv21_v = (kp_coord_uv[:, 1]) * crop_scale
        coords_uv = np.stack([keypoint_uv21_u, keypoint_uv21_v], 1)
        return coords_uv


if __name__ == '__main__':

    #_, hand_image, pose2d, relative_depth, uv_vis, _, _, _, _, _ = train_dataset.__getitem__(7)
    # print(GHDDateset(type=0).__len__())
    # print(GHDDateset(type=1).__len__())
    # print(GHDDateset(type=2).__len__())
    train_dataset = GHDDateset()
    _, hand_image, pose2d, relative_depth, uv_vis, _, _, _, _, _, _, _ = train_dataset.__getitem__(5)
    hand_image = hand_image.permute(1,2,0).numpy()
    hand_image = (hand_image + 1)/2
    pose2d = pose2d.numpy()

    print(hand_image.shape)
    print(pose2d.shape)


    #
    hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
    print(hand_image.shape)

    print(uv_vis.shape)
    print(uv_vis)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(hand_image)
    plt.scatter(pose2d[:, 0], pose2d[:, 1], alpha=0.6)

    plt.show()