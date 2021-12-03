import numpy as np
class Constant:
    width, height = 640, 480
    # width, height = 1280, 720
    depth_element_byte = 2
    rgb_element_byte = 1
    depth_frame_length_in_byte = width * height * depth_element_byte
    rgb_frame_length_in_byte = width * height * 3 * rgb_element_byte
    depth_type = np.uint16
    rgb_type = np.uint8
    far_range = 2000.0
    near_range = 200.0


    average_f = 614.0

    depth_scale = 0.001
    #depth_scale = 0.000124987

    crop_size_3d = (270, 270, 350)# 10.23
    #crop_size_3d = (300, 300, 350)  # original

    #crop_size_3d = (300, 300, 300)# original
    crop_size_2d = (64, 64)# original


    # hand detection configurations
    detect_img_ratio = 0.25
    detect_width = int(width * detect_img_ratio)#160
    detect_height = int(height * detect_img_ratio)#120

    wristcolor=(255, 0, 0)
    indexcolor=(25, 255, 25)
    middlecolor=(212, 0, 255)
    ringcolor=(0, 230, 230)
    pinkycolor=(179, 179, 0)
    thumbcolor=(255, 153, 153)
    joint_color = [wristcolor] * 1 + \
                  [indexcolor] * 4 + \
                  [middlecolor] * 4 + \
                  [ringcolor] * 4 + \
                  [pinkycolor] * 4 + \
                  [thumbcolor] * 4
    mano_joints_color=np.array([wristcolor]+[indexcolor]*3+[middlecolor]*3+
                               [pinkycolor]*3+[ringcolor]*3+[thumbcolor]*4+\
                                [indexcolor,middlecolor,ringcolor,pinkycolor])
    finger_color=np.array([indexcolor]+[middlecolor]+[ringcolor]+[pinkycolor]+[thumbcolor])
