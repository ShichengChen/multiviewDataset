import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
from .constant import Constant
constant = Constant()
def fetch_depth_data_shape(depth_path: str):
    file_size_in_bytes = os.path.getsize(depth_path)
    #print('depth file_size_in_bytes',file_size_in_bytes)
    return (
        file_size_in_bytes // constant.depth_frame_length_in_byte,
        constant.height,
        constant.width,
        1)

def fetch_rgb_data_shape(rgb_path: str):

    file_size_in_bytes = os.path.getsize(rgb_path)
    #print('rgb_path file_size_in_bytes', file_size_in_bytes)
    return (
        file_size_in_bytes // constant.rgb_frame_length_in_byte,
        constant.height,
        constant.width,
        3)

def load_depth_maps(depth_path):
    depth_shape = fetch_depth_data_shape(depth_path)
    #print('depth_shape',depth_shape)
    return np.memmap(depth_path, dtype=constant.depth_type, mode='r', shape=depth_shape)
    #return np.memmap(depth_path, dtype=np.float16, mode='r', shape=depth_shape)

def load_depth_cali_maps(depth_path):
    file_size_in_bytes = os.path.getsize(depth_path)
    depth_frame_length_in_byte = 720 * 1280 * 2
    depth_shape = (file_size_in_bytes // depth_frame_length_in_byte,720,1280,1)
    return np.memmap(depth_path, dtype=constant.depth_type, mode='r', shape=depth_shape)

def load_rgb_maps(rgb_path):
    rgb_shape = fetch_rgb_data_shape(rgb_path)
    #print('rgb_shape', rgb_shape)
    return np.memmap(rgb_path, dtype=constant.rgb_type, mode='r', shape=rgb_shape)

def load_rgb_cali_maps(rgb_path):
    file_size_in_bytes = os.path.getsize(rgb_path)
    rgb_frame_length_in_byte = 720 * 1280 * 3
    rgb_shape = (file_size_in_bytes // rgb_frame_length_in_byte,720,1280,3)
    return np.memmap(rgb_path, dtype=constant.rgb_type, mode='r', shape=rgb_shape)

def visualize_depth_map(depth_image):
        vis_depth_image = depth_image.copy().astype(np.float32)
        vis_depth_image = vis_depth_image * 255 / constant.far_range
        vis_depth_image = vis_depth_image.astype(np.uint8)
        return vis_depth_image
        # return cv2.cvtColor(vis_depth_image, cv2.COLOR_GRAY2BGR)
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

def get_cameras_from_dir(data_dir, folder_name, sequence_name):
    regex = os.path.join(data_dir,
                        folder_name,
                        '%s_*_depth.bin'%(sequence_name))
    dirs = glob.glob(regex)

    def fetch_cam_series(path: str):
        sub_str = path.split('_')
        if 'calib' in path:
                return sub_str[-3]
        else:
                return sub_str[-2]
    return [fetch_cam_series(d) for d in dirs]

def get_sequence_names_from_dir(data_dir, folder_name):
        regex = os.path.join(
                data_dir,
                folder_name,
                '*depth.bin')
        dirs = glob.glob(regex)
        sequence = set()
        for path in dirs:
                name = os.path.basename(path)
                if 'calib' in path:
                        continue
                name = name.split('_')[0]
                sequence.add(name)
        return list(sequence)



def fetch_all_sequences(file_path):
    file_name = os.path.basename(file_path)
    file_name = file_name.split('_')[0]
    folder_name = os.path.dirname(file_path)
    rgb_reg = os.path.join(folder_name, file_name + '*_rgb.bin')
    depth_reg = os.path.join(folder_name, file_name + '*_depth.bin')
    rgb_paths = glob.glob(rgb_reg)
    depth_paths = glob.glob(depth_reg)
    rgb_paths.sort()
    depth_paths.sort()
    return rgb_paths, depth_paths
