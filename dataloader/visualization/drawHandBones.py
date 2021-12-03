import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
from cscPy.Const.const import *
from cscPy.globalCamera.constant import Constant
joint_color = [(255, 0, 0)] * 1 + \
              [(25, 255, 25)] * 4 + \
              [(212, 0, 255)] * 4 + \
              [(0, 230, 230)] * 4 + \
              [(179, 179, 0)] * 4 + \
              [(0, 0, 255)] * 4
linecolor=np.array([[25, 255, 25],[212, 0, 255],[0, 230, 230],[179, 179, 0],[0, 0, 255]])
#linecolor=['green','magenta','yellow','cyan','white']
linesg = [[0, 1, 2, 3, 17], [0, 4, 5, 6, 18], [0, 7, 8, 9, 20], [0, 10, 11, 12, 19], [0, 13, 14, 15, 16]]



def drawHandJoints(img,poseuv,order='mano'):
    lineidx=[0,1,2,3,17,  4,5,6,18,   7,8,9,20, 10,11,12,19,  13,14,15,16]
    preidx= [0,0,1,2,3,   0,4,5,6,    0,7,8,9,   0,10,11,12, 0,13,14,15]
    prei =  [0,0,1,2,     0,4,5,       0,7,8,    0,10,11,    0,13,14,15, 3,6,12,9]
    for i in range(21):
        #uv=pose2uvd[i,:2].detach().cpu().numpy().astype(int)
        color = Constant.mano_joints_color[i].astype(int)
        #print(color)
        img=cv2.circle(img,(poseuv[i,0],poseuv[i,1]),2,color.tolist(),-1)
        img=cv2.line(img,(poseuv[i,0],poseuv[i,1]),(poseuv[prei[i],0],poseuv[prei[i],1]),color.tolist(),thickness=1)

    return img


def drawHandJointsWithImgIn3D(img,uvd,order='mano'):
    assert order == 'mano'
    assert len(uvd.shape) == 2
    color = Constant.finger_color
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    uvd=uvd / 64 * 255
    print(uvd)

    img=drawHandJoints(img,uvd)
    x, y = np.ogrid[0:img.shape[0], 0:img.shape[1]]
    ax.plot_surface(y,x,np.atleast_2d(0), rstride=2, cstride=2, facecolors=img.astype('float32') / 255)

    for i in range(len(linesg)):
        pass
        ax.plot(uvd[linesg[i], 0], uvd[linesg[i], 1], uvd[linesg[i], 2], marker='o', color=color[i]/255)
    ax.set_xlim3d(-10, 260)
    ax.set_ylim3d(-10, 260)
    ax.set_zlim3d(-10, 260)


    ax.set_xlabel('x label')
    ax.set_ylabel('y label')
    ax.set_zlabel('z label')
    #plt.show()



