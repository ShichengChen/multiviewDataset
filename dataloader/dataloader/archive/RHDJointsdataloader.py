from __future__ import print_function, unicode_literals

import sys

import scipy.io
import scipy.misc
import cscPy.dataAugment.augment as augment
import torchvision
#sys.path.append('..')
import pickle
import os
from torch.utils.data import Dataset
from cscPy.mano.network.utils import *
from cscPy.mano.network.utilsSmallFunctions import *
from cscPy.globalCamera.camera import *
from cscPy.mano.network.manolayer import MANO_SMPL
from cscPy.mano.network.net import VPoser
import tqdm
from cscPy.Const.const import *
from cscPy.handProcess.preprocessImages import preprocessMVdataset,imcrop
import cv2


# SET THIS to where RHD is located on your machine
path_to_db = './RHD_published_v2/'
if(not os.path.exists(path_to_db)):
    path_to_db = '/mnt/data/shicheng/RHD_published_v2/'
if(not os.path.exists(path_to_db)):
    path_to_db = '/home/csc/dataset/RHD_published_v2/'
if (not os.path.exists(path_to_db)):
    path_to_db = '/mnt/ssd/csc/RHD_published_v2/'
if (not os.path.exists(path_to_db)):
    path_to_db = '/mnt/data/csc/RHD_published_v2/'
    #os.environ["DISPLAY"] = "localhost:11.0"

set = 'training'
set = 'evaluation'


def getDir(joint_gt):
    joints=joint_gt.copy()
    manoidx = [1, 2, 3, 17,  4, 5, 6, 18,  7, 8, 9, 20,  10,11, 12, 19, 13,14, 15, 16]
    manopdx = [0, 1, 2, 3,   0, 4, 5, 6,   0, 7, 8, 9,   0,10, 11, 12,   0,13, 14, 15]
    for idx in range(len(manoidx)):
        ci = manoidx[idx]
        pi = manopdx[idx]
        joints[ci] = joints[ci] - joints[pi]
    joints[0] = 0
    return joints
def getDis(joint_gt):
    joints=joint_gt.copy()
    manoidx = [1, 2, 3, 17,  4, 5, 6, 18,  7, 8, 9, 20,  10,11, 12, 19, 13,14, 15, 16]
    manopdx = [0, 1, 2, 3,   0, 4, 5, 6,   0, 7, 8, 9,   0,10, 11, 12,   0,13, 14, 15]
    for idx in range(len(manoidx)):
        ci = manoidx[idx]
        pi = manopdx[idx]
        joints[ci] = np.linalg.norm(joints[ci] - joints[pi])
    joints[0]=0
    return joints



def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 2**8 + bottom_bits).astype('float32')
    depth_map /= float(2**16 - 1)
    depth_map *= 5.0
    return depth_map
class RHDJointsDataloader(Dataset):
    def __init__(self, train=True,path_name=path_to_db):
        print("loading rhd")
        if(train):self.mode='training'
        else:self.mode='evaluation'
        self.train=train
        self.num_samples=0
        self.path_name=path_name
        self.datasetname = 'RHD'
        with open(os.path.join(self.path_name, self.mode, 'anno_%s.pickle' % self.mode), 'rb') as fi:
            self.anno_all = pickle.load(fi)
            self.num_samples = len(self.anno_all.items())
        print('RHDDateset3D self.num_samples',self.num_samples)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if (idx == 20500 or idx == 28140): idx = 0
        #image = scipy.misc.imread(os.path.join(path_to_db, self.mode, 'color', '%.5d.png' % idx))
        mask = scipy.misc.imread(os.path.join(path_to_db, self.mode, 'mask', '%.5d.png' % idx))
        #depth = scipy.misc.imread(os.path.join(path_to_db, self.mode, 'depth', '%.5d.png' % idx))
        #print('depth.shape', depth.shape)
        #depth = depth_two_uint8_to_float(depth[:, :, 0], depth[:, :, 1])
        anno=self.anno_all[idx]

        # kp_coord_uv = anno['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel
        kp_visible = anno['uv_vis'][:, 2] == 1  # visibility of the keypoints, boolean
        kp_coord_xyz = anno['xyz'].astype(np.float32).copy()  # x, y, z coordinates of the keypoints, in meters
        K = anno['K'].astype(np.float32)  # matrix containing intrinsic parameters

        kp_coord_uvd= perspectiveProjection(kp_coord_xyz,K)
        #print(kp_coord_xyz[0],kp_coord_uvd[0],K)
        k_ = np.linalg.inv(K)


        cond_l = np.logical_and(mask > 1, mask < 18)
        cond_r = mask > 17
        num_px_left_hand = np.sum(cond_l)
        num_px_right_hand = np.sum(cond_r)
        hand_side_left = num_px_left_hand > num_px_right_hand
        if hand_side_left:
            kp_coord_xyz = kp_coord_xyz[:21, :]
        else:
            kp_coord_xyz = kp_coord_xyz[-21:, :]

        if hand_side_left:
            kp_coord_uvd = kp_coord_uvd[:21, :]
            uv_vis = kp_visible[:21]
        else:
            kp_coord_uvd = kp_coord_uvd[-21:, :]
            uv_vis = kp_visible[-21:]
        # RHD2mano_skeidx=[0,8,7,6, 12,11,10, 20,19,18, 16,15,14, 4,3,2,1, 5,9,13,17]
        kp_coord_uvd=kp_coord_uvd[RHD2mano_skeidx].copy()
        uv_vis=uv_vis[RHD2mano_skeidx].copy()
        kp_coord_xyz=kp_coord_xyz[RHD2mano_skeidx].copy()

        if(hand_side_left):
           # image = cv2.flip(image, 1)
            kp_coord_xyz[:, 0] = -kp_coord_xyz[:, 0]
            #kp_coord_uvd[:,0]=image.shape[1]-kp_coord_uvd[:,0]


        assert np.sum(np.abs(kp_coord_uvd[:,-1]-kp_coord_xyz[:,-1]))<1e-4

        pose3d = kp_coord_xyz[:21, :]
        pose3d_root = pose3d[4:5, :].copy()
        pose3d_rel = pose3d - pose3d_root  # relative coords in metric coords
        #scaled = np.sqrt(np.sum((kp_coord_xyz[4, :] - kp_coord_xyz[5, :]) ** 2))
        scaled = 1
        pose3d = (pose3d_rel / scaled)

        if (self.train):
            _, pose3d = augment.processing_augmentation_RGB(None, pose3d)
            pose3d = np.reshape(pose3d, [21, 3])
        else:
            pose3d = np.reshape(pose3d, [21, 3])

        dic = {"pose_gt": get32fTensor(pose3d.reshape(21, 3)),
               "pose3d_root": get32fTensor(pose3d_root.reshape(1, 3)),
               "scaled": get32fTensor(np.array([scaled]).reshape(1, 1)),
               }
        return dic



if __name__ == '__main__':

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    lr = 1e-3
    device = 'cuda'
    model=VPoser(inshape=21*2,dim=3).to(device)
    mylist=[]
    mylist.append({'params': model.parameters(), 'weight_decay': 1e-6})
    optimizer = torch.optim.Adam(mylist, lr=lr)

    def _init_fn(worker_id):
        np.random.seed(worker_id)

    train_dataset=RHDJointsDateset(mode=set,righthand=True)
    #train_dataset=STBDateset3D()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=False,
                                               worker_init_fn=_init_fn,)
    mano_right = MANO_SMPL('/home/csc/MANO-hand-model-toolkit/mano/models/MANO_'+"RIGHT.pkl" if train_dataset.righthand else 'LEFT.pkl', ncomps=45,
                           bighandorder=False)
    minn = 5
    allepochs = 100
    n2m = 1
    vertexout=[]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=allepochs // 3, gamma=0.1)
    for epoch in tqdm.tqdm(range(allepochs)):
        aveloss, aveloss2 = [], []
        model.eval()
        cnt = 0
        if(epoch==71):break

        for idx, (out) in enumerate(train_loader):
            cnt += 1
            jd = out['f'].to(device)
            n = jd.shape[0]
            # print(jd.shape)
            joints_gt=jd[:,0,...]
            # print('joints_gt.shape',joints_gt.shape)
            jd=jd.reshape(n,-1)
            scale = out['scale'].to(device)
            joint_root = out['root'].to(device)

            # print('jd',jd)
            # print('joints_gt',joints_gt)

            results = model(jd)

            vertex, joint_pre = \
                mano_right.get_mano_vertices(results['pose_aa'][:, 0:1, :],
                                             results['pose_aa'][:, 1:, :], results['shape'],
                                             results['scale'], results['transition'],
                                             pose_type='euler', mmcp_center=False)
            if (epoch==70):
                np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
                np_rot_x = np.reshape(np.tile(np_rot_x, [1, 1]), [1, 3, 3])
                vertex,scale,joint_root=vertex.reshape(n,778,3),scale.reshape(n,1,1),joint_root.reshape(n,1,3)
                #vertices = (vertex * scale + joint_root)[0].cpu().detach().numpy()
                vertices = (vertex * scale + joint_root)[0].cpu().detach().numpy()
                vertexout.append((vertex * scale + joint_root).cpu().detach().numpy())

                # faces = mano_right.faces
                # import trimesh
                # mesh = trimesh.Trimesh(vertices=vertices, faces=mano_right.faces)
                # mesh.visual.vertex_colors = [.9, .7, .7, 1]
                # mesh.show()

            # print('joint_pre',joint_pre)
            # print('joints_gt',joints_gt)

            joint_pre = joint_pre.view([n, 21, 3])
            joints_gt = joints_gt.reshape([n, 21, 3])
            jointdif = (torch.sqrt(torch.sum((joint_pre - joints_gt) ** 2, dim=2))).view(n, 21)
            eucDistance = jointdif * scale.view(n, 1)

            dloss = torch.mean(eucDistance)

            aveloss2.append(float(torch.mean(jointdif[:, 1:] * scale.view(n, 1))))

            pose_regular = torch.mean(torch.norm((results['pose_aa'][:, 1:, :]).reshape(-1, 3), dim=1)) * 1e-2 + \
                           torch.mean(torch.norm((results['pose_aa'][:, 0:1, :]).reshape(-1, 3), dim=1)) * 1e-3

            shape_regular = torch.mean(torch.sum(results['shape'] ** 2, dim=1)) * 1e-4

            loss = dloss + pose_regular + shape_regular
            aveloss.append(float(dloss))
            if (idx % 200 == 0):
                print('epoch:{} iteration:{} loss:{:.2f}'.
                      format(epoch, idx, loss.item() * n2m))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # out=invAdjust(out)
        scheduler.step()
        if (np.mean(aveloss) * n2m < minn):
            savename = 'iknet.pt'
            print('save model', savename)

            minn = min(minn, np.mean(aveloss) * n2m)
            torch.save({
                'epoch': epoch,
                'epe': minn * n2m,
                'iknet': model.state_dict(),
                'optimizer': optimizer.state_dict()}, savename)
        minn = min(minn, np.mean(aveloss) * n2m)
        print("ave aveloss:", np.mean(aveloss) * n2m, 'minn:', minn, 'no wrist', np.mean(aveloss2) * n2m)

    out=np.concatenate(vertexout,axis=0)
    print(out.shape)
    with open('rhdTest'+"RIGHTHand.npy" if train_dataset.righthand else 'LEFTHand.npy', 'wb') as f:
        np.save(f, out)



