import cv2
from .constant import Constant
import numpy as np

joint_color = [(255, 0, 0)] * 1 + \
              [(25, 255, 25)] * 4 + \
              [(212, 0, 255)] * 4 + \
              [(0, 230, 230)] * 4 + \
              [(179, 179, 0)] * 4 + \
              [(0, 0, 255)] * 4
linecolor=np.array([[25, 255, 25],[212, 0, 255],[0, 230, 230],[179, 179, 0],[0, 0, 255]])
linesg = [[0, 1, 2, 3, 17], [0, 4, 5, 6, 18], [0, 7, 8, 9, 20], [0, 10, 11, 12, 19], [0, 13, 14, 15, 16]]
epsilon=1e-6
constant = Constant()

def visualize_better_qulity_depth_map(depth_image):
    vis_depth_image = depth_image.copy().astype(np.float32)
    vis_depth_image = np.clip(vis_depth_image,0,2000)
    vis_depth_image = vis_depth_image * 255 / 2000
    vis_depth_image[vis_depth_image < 10] = 255
    vis_depth_image=vis_depth_image*3-50
    # mask=(vis_depth_image>60)&(vis_depth_image<100)
    # vis_depth_image[mask]=vis_depth_image[mask]*3-50
    # vis_depth_image[~mask]*=2
    #print(vis_depth_image)
    vis_depth_image = np.clip(vis_depth_image, 0, 255)
    vis_depth_image = vis_depth_image.astype(np.uint8)

    return cv2.cvtColor(vis_depth_image, cv2.COLOR_GRAY2BGR)



