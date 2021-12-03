# Python 2 Compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import math
import numpy as np
import cv2
import torch

POINT_SIZE = 256
LenDiscrete=64
ImageSize=256

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    if(torch.is_tensor(angle)):
        N=point.shape[0]
        angle=angle.reshape(N)
        po=(point-origin).reshape(N,21,2,1)
        rot=torch.zeros([N,1,2,2],dtype=angle.dtype,device=angle.device)
        rot[:,0,0,0]=torch.cos(angle)
        rot[:,0,1,1]=torch.cos(angle)
        rot[:,0,0,1]=-torch.sin(angle)
        rot[:,0,1,0]=torch.sin(angle)
        #rot=torch.tensor([[torch.cos(angle),-torch.sin(angle)],[torch.sin(angle),torch.cos(angle)]]).reshape(N,1,2,2)
        return (rot@po).reshape(N,21,2)+origin.reshape(1,1,2)
    else:
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

        return qx, qy

 
def invertAugmentation(pose3d,dic,mask):
    device,N=pose3d.device,pose3d.shape[0]
    randTrans, randScale, randAngle=dic['randTrans'].to(device),dic['randScale'].to(device),dic['randAngle'].to(device)
    if mask is not None:
        randTrans,randScale,randAngle=randTrans[mask],randScale[mask],randAngle[mask]
    pose3d = pose3d / LenDiscrete * ImageSize
    pose3d=pose3d.view(N,21,3)
    centerpoints=torch.tensor([ImageSize//2,ImageSize//2],dtype=torch.float32,device=device).reshape(1,1,2)
    pose3d[:,:,:2] -= randTrans
    pose3d[:,:,:2] = (pose3d[:,:,:2] - centerpoints) / randScale + centerpoints
    pose3d[:,:,:2]=rotate(centerpoints,pose3d[:,:,:2],angle=-randAngle)
    pose3d = pose3d / ImageSize * LenDiscrete
    return pose3d


# def processing_augmentation_joints(pose3d):
#     randScale = np.random.uniform(low=0.8, high=1.0)
#     pose3d = np.reshape(pose3d, [21, 3])
#     randAngle = 2 * math.pi * np.random.rand(1)[0]
#     rotMat = cv2.getRotationMatrix2D((ImageSize // 2, ImageSize // 2), -180.0 * randAngle / math.pi,
#                                      randScale)  # change image later together with translation
#     (pose3d[:, 0], pose3d[:, 1]) = rotate((0, 0), (pose3d[:, 0], pose3d[:, 1]), randAngle)
#     randTransX = np.maximum(np.minimum(np.random.normal(0.0, 22.0), 40.0), -40.0)
#     randTransY = np.maximum(np.minimum(np.random.normal(0.0, 22.0), 40.0), -40.0)
#     rotMat[0, 2] += randTransX
#     rotMat[1, 2] += randTransY
#     pose3d = np.reshape(pose3d, [63])
#     return pose3d

def rgb_processing(rgb_img):
    # in the rgb image we add pixel noise in a channel-wise manner
    noise_factor = 0.4
    pn = np.random.uniform(1 - noise_factor, 1 + noise_factor, 3)
    rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
    rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
    rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))
    return rgb_img
def processing_augmentation_Heatmap(image,pose3d,ImageSize=256):
    pose3d = pose3d / LenDiscrete * ImageSize
    randScale = np.random.uniform(low=0.8, high=1.0)
    #randScale = np.random.uniform(low=1.0, high=1.0)
    pose3d = np.reshape(pose3d, [21, 3])
    randAngle = 2 * math.pi * np.random.rand(1)[0]
    rotMat = cv2.getRotationMatrix2D((ImageSize//2, ImageSize//2), -180.0 * randAngle / math.pi,randScale)
    centerpoints=np.array([ImageSize//2,ImageSize//2])
    (pose3d[:, 0],pose3d[:, 1]) = rotate(centerpoints, (pose3d[:, 0], pose3d[:, 1]), randAngle)
    (pose3d[:, :2])=(pose3d[:, :2]-centerpoints)*randScale+centerpoints
    randTransU = np.maximum(np.minimum(np.random.normal(0.0, 22.0), 40.0), -40.0)
    randTransV = np.maximum(np.minimum(np.random.normal(0.0, 22.0), 40.0), -40.0)
    rotMat[0, 2] += randTransU
    rotMat[1, 2] += randTransV
    randTrans=np.array([randTransU,randTransV])
    pose3d[:, :2]+=randTrans
    image = cv2.warpAffine(image, rotMat, (ImageSize, ImageSize), flags=cv2.INTER_NEAREST, borderValue=0.0)
    pose3d = np.reshape(pose3d, [63])
    image = np.reshape(image, [ImageSize, ImageSize, 3])

    # sub = np.array([123.68, 116.779, 103.939]).reshape(1, 1, 3)
    # div = np.array([[58.393, 57.12, 57.375]]).reshape(1, 1, 3)
    #image=(rgb_processing(image)-sub)/div
    # image=rgb_processing(image)

    '''
    Normalize RGB channels by subtracting 123.68,
    116.779, 103.939 and dividing by 58.393, 57.12,
    57.375, respectively.
    '''

    pose3d = pose3d / ImageSize * LenDiscrete

    return image, pose3d,randTrans.reshape(1,2),np.array([[randScale]]),np.array([randAngle])


def processing_augmentation_RGB(image,pose3d):
    randScale = np.random.uniform(low=0.8, high=1.0)
    pose3d = np.reshape(pose3d, [21, 3])
    randAngle = 2 * math.pi * np.random.rand(1)[0]
    rotMat = cv2.getRotationMatrix2D((ImageSize//2, ImageSize//2), -180.0 * randAngle / math.pi,
                                     randScale)  # change image later together with translation
    (pose3d[:, 0], pose3d[:, 1]) = rotate((0, 0), (pose3d[:, 0], pose3d[:, 1]), randAngle)
    randTransX = np.maximum(np.minimum(np.random.normal(0.0, 22.0), 40.0), -40.0)
    randTransY = np.maximum(np.minimum(np.random.normal(0.0, 22.0), 40.0), -40.0)
    rotMat[0, 2] += randTransX
    rotMat[1, 2] += randTransY
    pose3d = np.reshape(pose3d, [63])
    if(image is not None):
        image = cv2.warpAffine(image, rotMat, (ImageSize, ImageSize), flags=cv2.INTER_NEAREST, borderValue=0.0)
        image = np.reshape(image, [ImageSize, ImageSize, 3])
    return image, pose3d

def processing_rgbCloud(image, cloud, pose3d):
    randScale = np.random.uniform(low=0.8, high=1.0)
    randScaleCloud = np.random.uniform(low=0.8, high=1.2)
    #cloud = cloud*randScaleCloud

    # Random rotation around z-axis (must be same rotation for image and pointcloud, because it affects the normalized pose)
    pose3d = np.reshape(pose3d, [21, 3])
    randAngle = 2 * math.pi * np.random.rand(1)[0]
    rotMat = cv2.getRotationMatrix2D((ImageSize//2, ImageSize//2), -180.0 * randAngle / math.pi,
                                     randScale)  # change image later together with translation
    (cloud[:, 0], cloud[:, 1]) = rotate((0, 0), (cloud[:, 0], cloud[:, 1]), randAngle)
    (pose3d[:, 0], pose3d[:, 1]) = rotate((0, 0), (pose3d[:, 0], pose3d[:, 1]), randAngle)

    # # Random translation (can be different tranlsations for image and pointcloud, because doesn't affect the normalized pose)
    randTransX = np.maximum(np.minimum(np.random.normal(0.0, 22.0), 40.0), -40.0)
    randTransY = np.maximum(np.minimum(np.random.normal(0.0, 22.0), 40.0), -40.0)

    rotMat[0, 2] += randTransX
    rotMat[1, 2] += randTransY
    image = cv2.warpAffine(image, rotMat, (ImageSize, ImageSize), flags=cv2.INTER_NEAREST, borderValue=0.0)

    # cloud = cloud + np.float32(np.maximum(np.minimum(np.random.normal(0.0, 0.1, (3,)),1.0),-1.0))

    randInidices = np.arange(len(cloud))
    np.random.shuffle(randInidices)
    cloud = cloud[randInidices[0:POINT_SIZE, ], :]


    pose3d = np.reshape(pose3d, [63])
    image = np.reshape(image, [ImageSize, ImageSize, 3])

    return image, cloud, pose3d


def processing_augmentation(image, depth, cloud, heatmap, pose3d, hand_side):
    # Random scaling (can be different scaling factors for image and pointclou, because doesn't affect the normalized pose)
    # randScale =  np.maximum(np.minimum(np.random.normal(1.0, 0.08),1.16),0.84)  #process image later together with warpAffine
    randScale = np.random.uniform(low=0.8, high=1.0)
    randScaleCloud = np.random.uniform(low=0.8, high=1.2)
    #cloud = cloud*randScaleCloud

    # Random rotation around z-axis (must be same rotation for image and pointcloud, because it affects the normalized pose)
    pose3d = np.reshape(pose3d, [21, 3])
    randAngle = 2 * math.pi * np.random.rand(1)[0]
    rotMat = cv2.getRotationMatrix2D((ImageSize//2, ImageSize//2), -180.0 * randAngle / math.pi,
                                     randScale)  # change image later together with translation
    rotMatHeatMap = cv2.getRotationMatrix2D((32, 32), -180.0 * randAngle / math.pi,
                                            randScale)  # change image later together with translation

    (cloud[:, 0], cloud[:, 1]) = rotate((0, 0), (cloud[:, 0], cloud[:, 1]), randAngle)
    (pose3d[:, 0], pose3d[:, 1]) = rotate((0, 0), (pose3d[:, 0], pose3d[:, 1]), randAngle)

    # # Random translation (can be different tranlsations for image and pointcloud, because doesn't affect the normalized pose)
    randTransX = np.maximum(np.minimum(np.random.normal(0.0, 22.0), 40.0), -40.0)
    randTransY = np.maximum(np.minimum(np.random.normal(0.0, 22.0), 40.0), -40.0)

    rotMat[0, 2] += randTransX
    rotMat[1, 2] += randTransY
    rotMatHeatMap[0, 2] += randTransX * 0.25
    rotMatHeatMap[1, 2] += randTransY * 0.25
    image = cv2.warpAffine(image, rotMat, (ImageSize, ImageSize), flags=cv2.INTER_NEAREST, borderValue=0.0)
    heatmap = cv2.warpAffine(heatmap, rotMatHeatMap, (LenDiscrete, LenDiscrete), flags=cv2.INTER_LINEAR, borderValue=0.0)
    depth = cv2.warpAffine(depth, rotMat, (ImageSize, ImageSize), flags=cv2.INTER_NEAREST, borderValue=10.0)

    # cloud = cloud + np.float32(np.maximum(np.minimum(np.random.normal(0.0, 0.1, (3,)),1.0),-1.0))

    randInidices = np.arange(len(cloud))
    np.random.shuffle(randInidices)
    #print(randInidices[:3])
    #print('randInidices',randInidices[:3])
    cloud = cloud[randInidices[0:POINT_SIZE, ], :]

    # flipping
    if (hand_side[0] == 0.0):
        image = cv2.flip(image, 1)
        depth = cv2.flip(depth, 1)
        heatmap = cv2.flip(heatmap, 1)
        cloud[:, 0] = -cloud[:, 0]
        pose3d[:, 0] = -pose3d[:, 0]

    pose3d = np.reshape(pose3d, [63])
    image = np.reshape(image, [ImageSize, ImageSize, 3])
    depth = np.reshape(depth, [ImageSize, ImageSize, 1])

    heatmap = np.reshape(heatmap, [LenDiscrete, LenDiscrete, 21])

    return image, depth, cloud, heatmap, pose3d


def processing(image, depth, cloud, heatmap, pose3d, hand_side):
    randInidices = np.arange(len(cloud))
    np.random.shuffle(randInidices)
    cloud = cloud[randInidices[0:POINT_SIZE, ], :]

    pose3d = np.reshape(pose3d, [21, 3])

    if (hand_side[0] == 0.0):
        image = cv2.flip(image, 1)
        depth = cv2.flip(depth, 1)
        cloud[:, 0] = -cloud[:, 0]
        pose3d[:, 0] = -pose3d[:, 0]

    pose3d = np.reshape(pose3d, [63])

    image = np.reshape(image, [ImageSize, ImageSize, 3])
    depth = np.reshape(depth, [ImageSize, ImageSize, 1])
    heatmap = np.reshape(heatmap, [LenDiscrete, LenDiscrete, 21])

    return image, depth, cloud, heatmap, pose3d
