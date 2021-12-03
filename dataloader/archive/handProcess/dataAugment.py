# Imports
import math
import numpy as np
import cv2
import torch

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy
def processing_augmentation(image, depth, cloud, pose3d, train, sampleN=906):
    if(train):
        # Random scaling (can be different scaling factors for image and pointclou, because doesn't affect the normalized pose)
        # randScaleImage =  np.maximum(np.minimum(np.random.normal(1.0, 0.08),1.16),0.84)  #process image later together with warpAffine
        randScaleImage = np.random.uniform(low=0.8, high=1.0)
        randScaleCloud = np.random.uniform(low=0.8, high=1.2)
        #cloud = cloud*randScaleCloud

        # Random rotation around z-axis (must be same rotation for image and pointcloud, because it affects the normalized pose)
        pose3d = np.reshape(pose3d, [21, 3])
        randAngle = 2 * math.pi * np.random.rand(1)[0]
        rotMat = cv2.getRotationMatrix2D((128, 128), -180.0 * randAngle / math.pi,
                                         randScaleImage)  # change image later together with translation

        (cloud[:, 0], cloud[:, 1]) = rotate((0, 0), (cloud[:, 0], cloud[:, 1]), randAngle)
        (pose3d[:, 0], pose3d[:, 1]) = rotate((0, 0), (pose3d[:, 0], pose3d[:, 1]), randAngle)

        # # Random translation (can be different tranlsations for image and pointcloud, because doesn't affect the normalized pose)
        randTransX = np.maximum(np.minimum(np.random.normal(0.0, 10.0), 20.0), -20.0)
        randTransY = np.maximum(np.minimum(np.random.normal(0.0, 10.0), 20.0), -20.0)

        rotMat[0, 2] += randTransX
        rotMat[1, 2] += randTransY
        if(image is not None):
            image = cv2.warpAffine(image, rotMat, (256, 256), flags=cv2.INTER_NEAREST, borderValue=0.0)
        if(depth is not None):
            depth = cv2.warpAffine(depth, rotMat, (256, 256), flags=cv2.INTER_NEAREST, borderValue=10.0)

        # cloud = cloud + np.float32(np.maximum(np.minimum(np.random.normal(0.0, 0.1, (3,)),1.0),-1.0))

    randInidices = torch.randperm(cloud.shape[0]).numpy()
    #print(randInidices[:3])
    #print('randInidices',randInidices[:3])
    cloud = cloud[randInidices, :]

    pose3d = np.reshape(pose3d, [63])
    if (image is not None):
        image = np.reshape(image, [256, 256, 3])
    if (depth is not None):
        depth = np.reshape(depth, [256, 256, 1])

    return image, depth, cloud, pose3d