import pickle

import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
# from dataloader.mano.network.utils import *
# from dataloader.mano.network.utilsSmallFunctions import *
# from dataloader.mano.network.Const import *
def minusHomoVectors(v0, v1):
    v = v0 - v1
    if (v.shape[-1] == 1):
        v[..., -1, 0] = 1
    else:
        v[..., -1] = 1
    return v

class MANO_SMPL(nn.Module):
    def __init__(self, mano_pkl_path, ncomps = 10, flat_hand_mean=False,cuda=True,device='cuda'):
        super(MANO_SMPL, self).__init__()
        self.userotJoints=False
        # Load the MANO_RIGHT.pkl
        with open(mano_pkl_path, 'rb') as f:
            model = pickle.load(f, encoding='latin1')

        faces_mano = np.array(model['f'], dtype=int)

        # Add new faces for the wrist part and let mano model waterproof
        # for MANO_RIGHT.pkl
        faces_addition = np.array([[38, 122, 92], [214, 79, 78], [239, 234, 122],
                        [122, 118, 239], [215, 108, 79], [279, 118, 117],
                        [117, 119, 279], [119, 108, 215], [120, 108, 119],
                        [119, 215, 279], [214, 215, 79], [118, 279, 239],
                        [121, 214, 78], [122, 234, 92]])
        self.faces = np.concatenate((faces_mano, faces_addition), axis=0)

        self.flat_hand_mean = flat_hand_mean

        self.is_cuda = (torch.cuda.is_available() and cuda and device=='cuda')

        np_v_template = np.array(model['v_template'], dtype=np.float)
        np_v_template = torch.from_numpy(np_v_template).float()
        #print('np_v_template',np_v_template.shape) #np_v_template torch.Size([778, 3])

        self.size = [np_v_template.shape[0], 3]
        np_shapedirs = np.array(model['shapedirs'], dtype=np.float)
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        #print('np_shapedirs',np_shapedirs.shape)#np_shapedirs (10, 2334)

        np_shapedirs = torch.from_numpy(np_shapedirs).float()

        # Adding new joints for the fingertips. Original MANO model provide only 16 skeleton joints.
        np_J_regressor = model['J_regressor'].T.toarray()
        np_J_addition = np.zeros((778, 5))
        np_J_addition[745][0] = 1
        np_J_addition[333][1] = 1
        np_J_addition[444][2] = 1
        np_J_addition[555][3] = 1
        np_J_addition[672][4] = 1
        np_J_regressor = np.concatenate((np_J_regressor, np_J_addition), axis=1)
        np_J_regressor = torch.from_numpy(np_J_regressor).float()

        np_hand_component = np.array(model['hands_components'], dtype=np.float)[:ncomps]
        np_hand_component = torch.from_numpy(np_hand_component).float()

        #print("np_hand_component",np_hand_component.shape)

        np_hand_mean = np.array(model['hands_mean'], dtype=np.float)[np.newaxis,:]
        if self.flat_hand_mean:
            np_hand_mean = np.zeros_like(np_hand_mean)
        np_hand_mean = torch.from_numpy(np_hand_mean).float()

        np_posedirs = np.array(model['posedirs'], dtype=np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        np_posedirs = torch.from_numpy(np_posedirs).float()

        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)
        #print('self.parents',self.parents)

        np_weights = np.array(model['weights'], dtype=np.float)
        vertex_count = np_weights.shape[0]
        vertex_component = np_weights.shape[1]
        np_weights = torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component)

        e3 = torch.eye(3).float()

        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
        np_rot_x = np.reshape(np.tile(np_rot_x, [1, 1]), [1, 3, 3])
        self.base_rot_mat_x = Variable(torch.from_numpy(np_rot_x).float())

        joint_x = torch.matmul(np_v_template[:, 0], np_J_regressor)
        joint_y = torch.matmul(np_v_template[:, 1], np_J_regressor)
        joint_z = torch.matmul(np_v_template[:, 2], np_J_regressor)
        self.tjoints = torch.stack([joint_x, joint_y, joint_z, torch.ones_like(joint_x)], dim=1).numpy()
        self.J = torch.stack([joint_x, joint_y, joint_z], dim=1).numpy()
        self.bJ=torch.tensor(self.J.reshape(1,21,3),dtype=torch.float32)




        if self.is_cuda:
            np_v_template = np_v_template.cuda()
            np_shapedirs = np_shapedirs.cuda()
            np_J_regressor = np_J_regressor.cuda()
            np_hand_component = np_hand_component.cuda()
            np_hand_mean = np_hand_mean.cuda()
            np_posedirs = np_posedirs.cuda()
            e3 = e3.cuda()
            np_weights = np_weights.cuda()
            self.base_rot_mat_x = self.base_rot_mat_x.cuda()

        '''
        np_hand_component torch.Size([45, 45])
        np_v_template torch.Size([778, 3])
        np_shapedirs torch.Size([10, 2334])
        np_J_regressor torch.Size([778, 21])
        np_hand_component torch.Size([45, 45])
        np_hand_mean torch.Size([1, 45])
        np_posedirs torch.Size([135, 2334])
        weight torch.Size([1, 778, 16])
        '''

        self.register_buffer('v_template', np_v_template)
        self.register_buffer('shapedirs', np_shapedirs)
        self.register_buffer('J_regressor', np_J_regressor)
        self.register_buffer('hands_comp', np_hand_component)
        self.register_buffer('hands_mean', np_hand_mean)
        self.register_buffer('posedirs', np_posedirs)
        self.register_buffer('e3', e3)
        self.register_buffer('weight', np_weights)


    def getTemplate(self,beta,zero_wrist=False):
        v_shaped = torch.matmul(beta*10, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)
        if(zero_wrist):J-=J[:,0:1,:].clone()
        return J





    def forward(self, beta, theta, wrist_euler, pose_type, get_skin=False,external_transition=None):
        assert pose_type in ['pca', 'euler', 'rot_matrix'], print('The type of pose input should be pca, euler or rot_matrix')
        num_batch = beta.shape[0]
        # print("num_batch",num_batch)

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)
        self.CJ=J.clone()
        #print("J.shape",J.shape)

        #global_rot = self.batch_rodrigues(wrist_euler).view(-1, 1, 3, 3)

        # pose_type should be 'pca' or 'euler' here
        if pose_type == 'pca':
            euler_pose = theta.mm(self.hands_comp) + self.hands_mean
            Rs = self.batch_rodrigues(euler_pose.contiguous().view(-1, 3))
            #print('Rs',Rs)
            global_rot = self.batch_rodrigues(wrist_euler.view(-1, 3)).view(-1, 1, 3, 3)
            #print("global_rot",global_rot)
        elif pose_type == 'euler':
            euler_pose = theta
            Rs = self.batch_rodrigues(euler_pose.contiguous().view(-1, 3)).view(-1, 15, 3, 3)
            global_rot = self.batch_rodrigues(wrist_euler.view(-1, 3)).view(-1, 1, 3, 3)
        else:
            Rs = theta.view(num_batch, 15, 3, 3)
            global_rot = wrist_euler.view(num_batch, 1, 3, 3)

        Rs = Rs.view(-1, 15, 3, 3)
        pose_feature = (Rs[:, :, :, :]).sub(1.0, self.e3).view(-1, 135)
        v_posed = v_shaped + torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])

        self.J_transformed, A,rotJoints = self.batch_global_rigid_transformation(torch.cat([global_rot, Rs], dim=1), J[:, :16, :], self.parents,JsAll=J.clone())

        weight = self.weight.repeat(num_batch, 1, 1)
        W = weight.view(num_batch, -1, 16)
        T = torch.matmul(W, A.view(num_batch, 16, 16)).view(num_batch, -1, 4, 4)

        ones_homo = torch.ones(num_batch, v_posed.shape[1], 1)
        if self.is_cuda:
            ones_homo = ones_homo.cuda()
        v_posed_homo = torch.cat([v_posed, ones_homo], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]
        if self.userotJoints:
            joints = rotJoints
        else:
            joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
            joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
            joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)
            joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        if get_skin:
            return verts, joints, Rs,
        else:
            return joints

    def get_mano_vertices(self, wrist_euler, pose, shape, scale, translation, pose_type = 'pca', mmcp_center = False,external_transition=None):
        """
        :param wrist_euler: mano wrist rotation params in euler representation [batch_size, 3]
        :param pose: mano articulation params [batch_size, 45] or pca pose [batch_size, ncomps]
        :param shape: mano shape params [batch_size, 10]
        :param cam: mano scale and translation params [batch_size, 3]
        :return: vertices: mano vertices Nx778x3,
                 joints: 3d joints in BigHand skeleton indexing Nx21x3
        """

        # apply parameters on the model
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float)
        if not isinstance(translation, torch.Tensor):
            translation = torch.tensor(translation, dtype=torch.float)
        if not isinstance(wrist_euler, torch.Tensor):
            wrist_euler = torch.tensor(wrist_euler, dtype=torch.float)
        if not isinstance(pose, torch.Tensor):
            pose = torch.tensor(pose, dtype=torch.float)
        if not isinstance(shape, torch.Tensor):
            shape = torch.tensor(shape, dtype=torch.float)

        if self.is_cuda:
            translation = translation.cuda()
            scale = scale.cuda()
            shape = shape.cuda()
            pose = pose.cuda()
            wrist_euler = wrist_euler.cuda()

        #
        if pose_type == 'pca':
            pose = pose.clamp(-2.,2.)
            #shape = shape.clamp(-0.03, 0.03)

        verts, joints, Rs = self.forward(shape, pose, wrist_euler, pose_type, get_skin=True,external_transition=external_transition)

        scale = scale.contiguous().view(-1, 1, 1)
        trans = translation.contiguous().view(-1, 1, 3)

        verts = scale * verts
        verts = trans + verts
        joints = scale * joints
        joints = trans + joints
        # mmcp is 3th joint in bighand order
        if mmcp_center:
            mmcp = joints[:, 3, :].clone().unsqueeze(1)
            verts -= mmcp
            joints -= mmcp

        #verts = torch.matmul(verts, self.base_rot_mat_x)

        joints = joints # convert to mm

        return verts, joints

    def quat2mat(self, quat):
        """Convert quaternion coefficients to rotation matrix.
        Args:
            quat: size = [B, 4] 4 <===>(w, x, y, z)
        Returns:
            Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
        """
        norm_quat = quat
        norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
        w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

        B = quat.size(0)

        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                              2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                              2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)

        return rotMat

    def batch_rodrigues(self, theta):
        l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
        angle = torch.unsqueeze(l1norm, -1)
        normalized = torch.div(theta, angle)
        angle = angle * 0.5
        v_cos = torch.cos(angle)
        v_sin = torch.sin(angle)
        quat = self.quat2mat(torch.cat([v_cos, v_sin * normalized], dim=1))
        return quat

    def batch_global_rigid_transformation(self, Rs, Js, parent,JsAll=None):
        N = Rs.shape[0]
        root_rotation = Rs[:, 0, :, :]
        Js = torch.unsqueeze(Js, -1)

        def make_A(R, t):
            R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
            ones_homo = Variable(torch.ones(N, 1, 1))
            if self.is_cuda:
                ones_homo = ones_homo.cuda()
            t_homo = torch.cat([t, ones_homo], dim=1)
            return torch.cat([R_homo, t_homo], 2)

        A0 = make_A(root_rotation, Js[:, 0])
        results = [A0]
        newjs=JsAll.clone().reshape(N,21,3)

        newjsones=torch.ones([N,21,1]).to(Rs.device)
        newjs=torch.cat([newjs,newjsones],dim=2).reshape(N,21,4,1)
        orijs = newjs.clone().reshape(N,21,4)
        transidx=[2,3,17,   5, 6, 18,    8, 9, 20,    11, 12, 19,   14, 15, 16]
        transpdx=[1,2,3,    4, 5, 6,     7, 8, 9,     10, 11, 12,   13, 14, 15]
        #manopdx=[1,2,3,   4,5, 6,    7,8, 9,    10,11, 12,   13,14, 15]
        #parent: 012 045 078 01011 01314
        cpidx=[1,4,7,10,13]
        for i in range(len(cpidx)):
            a=minusHomoVectors(orijs[:, cpidx[i]],orijs[:, 0]).reshape(N,4,1)
            newjs[:,cpidx[i]]=(A0@a)


        for i in range(1, parent.shape[0]):
            j_here = Js[:, i] - Js[:, parent[i]]
            A_here = make_A(Rs[:, i], j_here)
            res_here = torch.matmul(results[parent[i]], A_here)

            a = minusHomoVectors(orijs[:,transidx[i-1]], orijs[:,transpdx[i-1]]).reshape(N,4,1)
            newjs[:,transidx[i-1]]=(res_here@a)
            results.append(res_here)

        self.newjs=newjs
        results = torch.stack(results, dim=1)

        new_J = results[:, :, :3, 3] #did not use later
        ones_homo = Variable(torch.zeros(N, 16, 1, 1))
        if self.is_cuda:ones_homo = ones_homo.cuda()
        Js_w0 = torch.cat([Js, ones_homo], dim=2)
        init_bone = torch.matmul(results, Js_w0)
        init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
        A = results - init_bone

        return new_J, A, newjs.clone()[:,:,:-1,0]

