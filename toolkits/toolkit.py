import sys
sys.path.append("..")
sys.path.append(".")

from toolkits.manolayer import MANO_SMPL
from toolkits.globalCamera.util import visualize_better_qulity_depth_map
from toolkits.globalCamera.camera import CameraIntrinsics,perspective_projection,perspective_back_projection
from toolkits.globalCamera.constant import Constant
import os,pickle

import numpy as np
import torch
import cv2

def AxisRotMat(angles,rotation_axis):
    x,y,z=rotation_axis
    xx,xy,xz,yy,yz,zz=x*x,x*y,x*z,y*y,y*z,z*z
    c = np.cos(angles)
    s = np.sin(angles)
    i = 1 - c
    rot_mats=np.eye(4).astype(np.float32)
    rot_mats[0,0] =  xx * i + c
    rot_mats[0,1] =  xy * i -  z * s
    rot_mats[0,2] =  xz * i +  y * s

    rot_mats[1,0] =  xy * i +  z * s
    rot_mats[1,1] =  yy * i + c
    rot_mats[1,2] =  yz * i -  x * s

    rot_mats[2,0] =  xz * i -  y * s
    rot_mats[2,1] =  yz * i +  x * s
    rot_mats[2,2] =  zz * i + c
    rot_mats[3,3]=1
    return rot_mats
class MultiviewDatasetDemo():
    def __init__(self,manoPath,
                 file_path,
                 loadManoParam=False,
    ):
        if(manoPath==None):
            self.mano_right = None
        else:
            self.mano_right = MANO_SMPL(manoPath, ncomps=45)
        self.loadManoParam=loadManoParam
        self.readNotFromBinary=True
        baseDir = file_path
        self.baseDir=baseDir
        self.date = baseDir[baseDir.rfind('/') + 1:]
        calib_path = os.path.join(baseDir, 'calib.pkl')
        print("baseDir", baseDir)
        with open(calib_path, 'rb') as f:
            camera_pose_map = pickle.load(f)

        cam_list = ['840412062035','840412062037','840412062038','840412062076']
        self.cam_list = cam_list
        cam_list.sort() 
        camera, camera_pose,Ks = [], [],[]
        for camera_ser in cam_list:
            camera.append(CameraIntrinsics[camera_ser])
            camera_pose.append(camera_pose_map[camera_ser])
            K = np.eye(3)
            K[0, 0] = CameraIntrinsics[camera_ser].fx
            K[1, 1] = CameraIntrinsics[camera_ser].fy
            K[0, 2] = CameraIntrinsics[camera_ser].cx
            K[1, 2] = CameraIntrinsics[camera_ser].cy
            Ks.append(K.copy())
        for i in range(4):
            if (np.allclose(camera_pose[i], np.eye(4))):
                rootcameraidx = i
        # print("camera_pose",camera_pose)
        self.camera_pose,self.camera,self.Ks=camera_pose,camera,Ks
        self.rootcameraidx=rootcameraidx
        print('self.rootcameraidx',self.rootcameraidx)

        joints = np.load(os.path.join(baseDir, "mlresults", self.date + '-joints.npy'))
        self.N=joints.shape[0]
        self.joints=joints.reshape(self.N,21,4,1).astype(np.float32)

        joints4view = np.ones((4, self.N, 21, 4, 1)).astype(np.int64)
        for dev_idx, rs_dev in enumerate(cam_list):
            inv = np.linalg.inv(camera_pose[dev_idx])
            joints4view[dev_idx] = inv @ self.joints
        self.joints4view=joints4view

        self.avemmcp=np.mean(joints4view[rootcameraidx,:,5,:3],axis=0)

        if(loadManoParam):
            with open(os.path.join(self.baseDir,self.date+'manoParam.pkl'), 'rb') as f:
                self.manoparam = pickle.load(f)





    def getCameraIntrinsic(self,iv):
        cam_list = ['840412062035', '840412062037', '840412062038', '840412062076']
        return self.camera[cam_list[iv]]
    def getCameraPose(self):
        calib_path = os.path.join(self.baseDir, 'calib.pkl')
        with open(calib_path, 'rb') as f:
            camera_pose_map = pickle.load(f)
        cam_list = ['840412062035', '840412062037', '840412062038', '840412062076']
        camera_pose = []
        for camera_ser in cam_list:
            camera_pose.append(camera_pose_map[camera_ser])
        return camera_pose


    def readRGB(self,idx,iv):
        rgbpath = os.path.join(self.baseDir, 'rgb')
        rgbpath = os.path.join(rgbpath, "%05d" % (idx) + '_' + str(iv) + '.jpg')
        return cv2.imread(rgbpath)


    def getMask(self,idx,uselist=False):
        dms = []
        for iv in range(4): dms.append(self.readMask(idx, iv))
        if (uselist):
            return dms
        else:
            return np.hstack(dms)

    def readMask(self,idx,iv):
        rgbpath = os.path.join(self.baseDir, 'mask')
        rgbpath = os.path.join(rgbpath, "%05d" % (idx) + '_' + str(iv) + '.jpg')
        return cv2.imread(rgbpath)

    def decodeDepth(self,rgb:np.ndarray):
        """ Converts a RGB-coded depth into depth. """
        assert (rgb.dtype==np.uint8)
        r, g, _ = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        depth = (r.astype(np.uint64) + g.astype(np.uint64) * 256).astype(np.uint16)
        return depth

    def readDepth(self,idx,iv):
        dpath = os.path.join(self.baseDir, 'depth')
        dpath = os.path.join(dpath, "%05d" % (idx) + '_' + str(iv) + '.png')
        return self.decodeDepth(cv2.imread(dpath))

    def getImgs(self,idx,uselist=False):
        if(uselist):return self.getImgsList(idx)
        color=[]
        for iv in range(4):color.append(self.readRGB(idx,iv))
        return np.hstack(color)
    def getImgsList(self,idx,facemask=True):
        color=[]
        for iv in range(4):
            img=self.readRGB(idx,iv)
            color.append(img)
        return color
    def getDepth(self,idx,uselist=False):
        dms = []
        for iv in range(4): dms.append(self.readDepth(idx,iv))
        if(uselist):return dms
        else:return np.hstack(dms)
    def getBetterDepth(self,idx,uselist=False):
        dlist = []
        for iv in range(4):
            depth = self.readDepth(idx,iv)
            dlist.append(visualize_better_qulity_depth_map(depth))
        if (uselist): return dlist
        return np.hstack(dlist)

    def getManoVertex(self,idx):
        results,scale, joint_root = self.getManoParamFromDisk(idx)
        vertex, joint_pre = \
            self.mano_right.get_mano_vertices(results['pose_aa'][:, 0:1, :],
                                         results['pose_aa'][:, 1:, :], results['shape'],
                                         results['scale'], results['transition'],
                                         pose_type='euler', mmcp_center=False)
        vertex=vertex.cpu()
        scale=scale.cpu()
        joint_root=joint_root.cpu()
        vertices = (vertex * scale + joint_root)[0].cpu().detach().numpy() * 1000
        vertices = np.concatenate([vertices, np.ones([vertices.shape[0], 1])], axis=1)
        vertices = np.expand_dims(vertices, axis=-1)
        self.vertices=vertices
        return vertices

    def getManoParamFromDisk(self,idx):
        #return a dictionary which includes mano pose parameters, scale, and transition
        assert (self.loadManoParam==True)
        results=self.manoparam[idx]
        c=np.sqrt(np.sum((self.joints[idx,5,:3,0]/1000-self.joints[idx,1,:3,0]/1000)**2))
        scale=torch.tensor(c,dtype=torch.float32)
        joint_root=torch.tensor(self.joints[idx,5,:3,0]/1000,dtype=torch.float32)
        return results,scale, joint_root



    def get4viewManovertices(self,idx):
        vertices=self.getManoVertex(idx)
        vertices4view=np.zeros([4,778,4,1])
        for iv in range(4):
            inv = np.linalg.inv(self.camera_pose[iv])
            vertices4view[iv] = (inv @ vertices)
        self.vertices4view=vertices4view
        return vertices4view

    def render4mesh(self,idx,ratio=1):
        #the ratio=10 can make the rendered image be black
        vertices4view=self.get4viewManovertices(idx)
        import trimesh
        import pyrender
        from pyrender import RenderFlags
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        np_rot_x = np.reshape(np.tile(np_rot_x, [1, 1]), [3, 3])
        recolorlist=[]
        for iv in range(4):
            xyz=vertices4view[iv,:778,:3,0].copy()
            cv = xyz @ np_rot_x
            tmesh = trimesh.Trimesh(vertices=cv / 1000*ratio, faces=self.mano_right.faces)
            # tmesh.visual.vertex_colors = [.9, .7, .7, 1]
            # tmesh.show()
            mesh = pyrender.Mesh.from_trimesh(tmesh)
            scene = pyrender.Scene()
            scene.add(mesh)
            pycamera = pyrender.IntrinsicsCamera(self.camera[iv].fx, self.camera[iv].fy, self.camera[iv].cx, self.camera[iv].cy, znear=0.0001,
                                                 zfar=3000)
            ccamera_pose = self.camera_pose[self.rootcameraidx]
            scene.add(pycamera, pose=ccamera_pose)
            light = pyrender.SpotLight(color=np.ones(3), intensity=2.0, innerConeAngle=np.pi / 16.0)
            # light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
            scene.add(light, pose=ccamera_pose)
            r = pyrender.OffscreenRenderer(640, 480)
            # flags = RenderFlags.SHADOWS_DIRECTIONAL
            recolor, depth = r.render(scene)
            # cv2.imshow("depth", depth)
            recolorlist.append(recolor[:, :, :3])
        meshcolor = np.hstack(recolorlist)
        return meshcolor


    def drawMesh(self,idx):
        recolor=self.render4mesh(idx)
        color=np.hstack(self.getImgsList(idx))
        recolor[recolor == 255] = color[recolor == 255]
        c = cv2.addWeighted(color, 0.1, recolor, 0.9, 0.0)
        return c

    def getPose2D(self,idx,view):
        ujoints = self.joints4view[view, idx, :21, :3, 0].copy()
        uvdlist=[]
        for jdx in range(21):
            rgbuvd = perspective_projection(ujoints[jdx], self.camera[view]).astype(int)[:2]
            uvdlist.append(rgbuvd)
        return np.array(uvdlist).reshape(21,2)#uvd array

    def getPose3D(self,idx,view):
        return self.joints4view[view, idx, :21, :3, 0].copy()

    def drawPose4view(self,idx,view=4):
        assert (view == 1 or view == 4), "only support 4 and 1 view"
        lineidx = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]
        uvdlist = []
        imgs=self.getImgsList(idx)
        for iv in range(4):
            ujoints=self.joints4view[iv,idx,:21,:3,0].copy()
            for jdx in range(21):
                rgbuvd = perspective_projection(ujoints[jdx], self.camera[iv]).astype(int)[:2]
                uvdlist.append(rgbuvd)

                color=np.array(Constant.joint_color[jdx]).astype(int)
                imgs[iv] = cv2.circle(imgs[iv], tuple(rgbuvd), 3, color.tolist(), -1)
                if (jdx in lineidx):
                    imgs[iv] = cv2.line(imgs[iv], tuple(rgbuvd), tuple(uvdlist[-2]), color.tolist(), thickness=2)
        if(view==1):imgs = imgs[0].copy()
        else:imgs = np.hstack(imgs)
        return imgs




if __name__ == "__main__":
    file_path1 = "/media/csc/Seagate Backup Plus Drive/dataset/7-14-1-2"
    file_path2 = "/media/csc/Seagate Backup Plus Drive/dataset/9-10-1-2"
    file_path3 = "/media/csc/Seagate Backup Plus Drive/dataset/9-17-1-2"
    file_path4 = '/media/csc/Seagate Backup Plus Drive/dataset/9-25-1-2'
    file_paths = [file_path1,file_path2, file_path3, file_path4]
    manoPath = '/home/csc/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl'
    for path in file_paths:
        demo=MultiviewDatasetDemo(loadManoParam=True,file_path=path,manoPath=manoPath)
        for i in range(0,20):
            meshcolor=demo.drawMesh(i)
            cv2.imshow("meshcolor", meshcolor)
            imgs=demo.drawPose4view(i)
            cv2.imshow("imgs", imgs)
            depth = demo.getBetterDepth(i)
            cv2.imshow("depth", depth)
            cv2.waitKey(1)








