import numpy as np
import cv2

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    if img.ndim < 3: # for depth
        borderValue = [0]
    else: # for rgb
        # borderValue = [127, 127, 127]
        borderValue = [0, 0, 0]

    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                                 -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_CONSTANT, value=borderValue)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2
def imcrop(img, center, crop_size):
    x1 = int(np.round(center[0]-crop_size))
    y1 = int(np.round(center[1]-crop_size))
    x2 = int(np.round(center[0]+crop_size))
    y2 = int(np.round(center[1]+crop_size))

    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
         img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)

    if img.ndim < 3: # for depth
        img_crop = img[y1:y2, x1:x2]
    else: # for rgb
        img_crop = img[y1:y2, x1:x2, :]

    return img_crop,(x1, y1, x2, y2)

def imcrop2(img, x1,x2,y1,y2):
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
         img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    if img.ndim < 3: # for depth
        img_crop = img[y1:y2, x1:x2]
    else: # for rgb
        img_crop = img[y1:y2, x1:x2, :]
    return img_crop,(x1, y1, x2, y2)

def preprocessMVdataset(image, pose_uv_all, pose3d,cloud,cidx,cnxt,ratio=1.7):
    pose3d_root=pose3d[cidx:cidx+1]
    pose3d_rel = pose3d - pose3d_root  # relative coords in metric coords
    index_root_bone_length = np.sqrt(np.sum(np.square(pose3d_rel[cidx, :] - pose3d_rel[cnxt, :])))
    scale = index_root_bone_length
    pose3d_normed = pose3d_rel / scale
    if(cloud is not None):
        cloud_normed=(cloud-pose3d_root)/scale

    crop_center = pose_uv_all[cidx, :]
    crop_center = np.reshape(crop_center,2)
    crop_size = np.max(np.absolute(pose_uv_all-crop_center))*ratio
    crop_size = np.minimum(np.maximum(crop_size, 25.0), 200.0)
    image_crop,(x1, x2, y1, y2) = imcrop(image, crop_center, crop_size)
    image_crop = cv2.resize(image_crop, (256, 256), interpolation=cv2.INTER_NEAREST)

    if(cloud is None):
        return image_crop, pose3d_normed, np.array(scale).reshape(1, 1), pose3d_root.reshape(1, 3)
    else:
        return image_crop,pose3d_normed,cloud_normed,np.array(scale).reshape(1,1),pose3d_root.reshape(1,3)