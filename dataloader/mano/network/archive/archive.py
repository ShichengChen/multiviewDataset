def matchTemplate2JointsGreedy(self, joint_gt: np.ndarray, tempJ=None, restrainFingerDOF=0):
    # restrainFingerDOF=1 forward
    # restrainFingerDOF=2 backward
    # restrainFingerDOF=3 backward+finger angle
    N = joint_gt.shape[0]
    joint_gt = joint_gt.reshape(N, 21, 3)
    if (not torch.is_tensor(joint_gt)):
        joint_gt = torch.tensor(joint_gt, device='cpu', dtype=torch.float32)
    device = joint_gt.device

    # first make wrist to zero

    orijoint_gt = joint_gt.clone()
    oriWrist = orijoint_gt[:, 0:1, :].clone()
    joint_gt = joint_gt - oriWrist.clone()

    transformG = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 1, 4, 4).repeat(N, 16, 1,
                                                                                             1).reshape(N,
                                                                                                        16, 4, 4)
    transformL = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 1, 4, 4).repeat(N, 16, 1,
                                                                                             1).reshape(N,
                                                                                                        16, 4, 4)
    transformLmano = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 1, 4, 4).repeat(N, 16, 1,
                                                                                                 1).reshape(N,
                                                                                                            16, 4, 4)
    transformG[:, 0, :3, 3] = joint_gt[:, 0].clone()
    transformL[:, 0, :3, 3] = joint_gt[:, 0].clone()
    transformLmano[:, 0, :3, 3] = joint_gt[:, 0].clone()

    if (tempJ is None):
        tempJ = self.bJ.reshape(1, 21, 3).clone().repeat(N, 1, 1).reshape(N, 21, 3).to(device)
    else:
        # print("use external template")
        if (not torch.is_tensor(tempJ)): tempJ = torch.tensor(tempJ, dtype=torch.float32, device=device)
        if (len(tempJ.shape) == 3):
            tempJ = tempJ.reshape(N, 21, 3)
        else:
            tempJ = tempJ.reshape(1, 21, 3).clone().repeat(N, 1, 1).reshape(N, 21, 3)
    tempJori = tempJ.clone()
    tempJ = tempJ - tempJori[:, 0:1, :]
    tempJori = tempJori - tempJori[:, 0:1, :].clone()

    R = wristRotTorch(tempJ, joint_gt)
    transformG[:, 0, :3, :3] = R
    transformL[:, 0, :3, :3] = R
    transformLmano[:, 0, :3, :3] = R

    # print(joint_gt,tempJ)
    assert (torch.sum(joint_gt[:, 0] - tempJ[:, 0]) < 1e-5), "wrist joint should be same!" + str(
        torch.sum(joint_gt[:, 0] - tempJ[:, 0]))

    childern = [[1, 2, 3, 17, 4, 5, 6, 18, 7, 8, 9, 20, 10, 11, 12, 19, 13, 14, 15, 16],
                [2, 3, 17], [3, 17], [17],
                [5, 6, 18], [6, 18], [18],
                [8, 9, 20], [9, 20], [20],
                [11, 12, 19], [12, 19], [19],
                [14, 15, 16], [15, 16], [16]]

    for child in childern[0]:
        t1 = (tempJ[:, child] - tempJ[:, 0]).reshape(N, 3, 1)
        tempJ[:, child] = (transformL[:, 0] @ getHomo3D(t1)).reshape(N, 4, 1)[:, :-1, 0]

    if restrainFingerDOF == 0:
        pass
    elif restrainFingerDOF == 1:
        palmNorm = unit_vector(torch.cross(tempJ[:, 4] - tempJ[:, 0], tempJ[:, 7] - tempJ[:, 4], dim=1))
        palmd = -torch.sum((tempJ[:, 0] * palmNorm).reshape(N, 3), dim=1, keepdim=True)
        # palmPlane=torch.cat([palmNorm,palmd],dim=1).reshape(N,4)
        palmHorizon = unit_vector(tempJ[:, 1] - tempJ[:, 7])
        self.mcpjoints = joint_gt.clone()
    elif restrainFingerDOF == 2:
        joint_gt = self.restrainFingerDirectly(joint_gt)
    elif restrainFingerDOF == 3:
        # have both restrainFingerDirectly and finger angle
        joint_gt, loss = self.restrainFingerAngle(joint_gt)
    else:
        assert False, "wrong restrainFingerDOF"
    # cpidx = [0, 1, 4, 7, 10, 13]
    # for i in range(len(cpidx)):
    #     print("mcp dis", cpidx[i], torch.mean(torch.norm(tempJ[:,cpidx[i]] - joint_gt[:,cpidx[i]],dim=1)) * 1000)

    manoidx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
    manopdx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    manoppx = [0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
    jidx = [[0], [1, 2, 3, 17], [4, 5, 6, 18], [10, 11, 12, 19], [7, 8, 9, 20], [13, 14, 15, 16]]
    mcpidx = [1, 4, 10, 7]
    ratio = []
    for idx, i in enumerate(manoidx):
        pi = manopdx[idx]
        v0 = (tempJ[:, i] - tempJ[:, pi]).reshape(N, 3)
        v1 = (joint_gt[:, i] - tempJ[:, pi]).reshape(N, 3)

        # print('ratio',pi,i,torch.mean(torch.norm(v0)/torch.norm(joint_gt[:,i]-joint_gt[:,pi])))
        # ratio.append(np.linalg.norm(v0) / np.linalg.norm(v1))

        if (pi in mcpidx and restrainFingerDOF == 1):
            dis, projectedPoint = projectPoint2Plane(points=joint_gt[:, i], planeNorm=palmNorm, planeD=palmd)

            self.mcpjoints[:, i] = projectedPoint.clone().reshape(N, 3)
            vp = (projectedPoint - tempJ[:, pi]).reshape(N, 3)
            mask = (torch.norm(vp, dim=1) > torch.norm(v0, dim=1) * 0.5)
            N2 = torch.sum(mask)
            if (N2 > 0):
                pr = getRotationBetweenTwoVector(v0[mask], vp[mask])
                rotedHarizon = (pr.reshape(N2, 3, 3) @ palmHorizon[mask].reshape(N2, 3, 1))
                assert rotedHarizon.shape == (N2, 3, 1)
                fingerNorm = rotedHarizon.reshape(N2, 3)
                # fingerbaseD=v1.clone()
                # fingerNorm=unit_vector(torch.cross(rotedHarizon,fingerbaseD,dim=1).reshape(N,3))
                FingerD = -torch.sum(fingerNorm * tempJ[mask, i], dim=1, keepdim=True)
                dis1, projectedjoint1 = projectPoint2Plane(points=joint_gt[mask, manoidx[idx + 1]],
                                                           planeNorm=fingerNorm, planeD=FingerD)
                dis2, projectedjoint2 = projectPoint2Plane(points=joint_gt[mask, manoidx[idx + 2]],
                                                           planeNorm=fingerNorm, planeD=FingerD)
                joint_gt[mask, manoidx[idx + 1]] = projectedjoint1
                joint_gt[mask, manoidx[idx + 2]] = projectedjoint2

        tr = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 4, 4).repeat(N, 1, 1)
        r = getRotationBetweenTwoVector(v0, v1)
        tr[:, :3, :3] = r.clone()
        t0 = (tempJ[:, pi]).reshape(N, 3)
        tr[:, :-1, -1] = t0

        # print('r',r)

        transformL[:, idx + 1] = tr
        Gp = transformG[:, self.parents[idx + 1]].reshape(N, 4, 4)
        transformG[:, idx + 1] = transformL[:, idx + 1] @ Gp
        transformLmano[:, idx + 1] = torch.inverse(Gp) @ transformL[:, idx + 1] @ Gp

        for child in childern[pi]:
            t1 = (tempJ[:, child] - tempJ[:, pi]).reshape(N, 3, 1)
            tempJ[:, child] = (transformL[:, idx + 1] @ getHomo3D(t1)).reshape(N, 4, 1)[:, :-1, 0]

    local_trans = transformLmano[:, 1:, :3, :3].reshape(N, 15, 3, 3)
    wrist_trans = transformLmano[:, 0, :3, :3].reshape(N, 1, 3, 3)

    if (restrainFingerDOF == 1):
        self.newjoints = joint_gt.clone()

    outjoints = rotate2joint(wrist_trans, local_trans, tempJori, self.parents).reshape(N, 21, 3)
    assert (torch.mean(
        torch.sqrt(torch.sum((outjoints - tempJ) ** 2, dim=2))) < 2), "outjoints and tempJ epe should be small" + str(
        torch.mean(torch.sqrt(torch.sum((outjoints - tempJ) ** 2, dim=2))))
    # print(torch.mean(torch.sqrt(torch.sum((outjoints - tempJ) ** 2, dim=2))))

    outjoints = outjoints + oriWrist
    # print('oriWrist',oriWrist)
    # print('innr edu',torch.mean(torch.sqrt(torch.sum((outjoints-orijoint_gt)**2,dim=2))))
    # print('innr2 edu',torch.mean(torch.sqrt(torch.sum(((outjoints+oriWrist)-(orijoint_gt))**2,dim=2))))

    if (restrainFingerDOF == 3):
        return wrist_trans, local_trans, outjoints, loss
    return wrist_trans, local_trans, outjoints