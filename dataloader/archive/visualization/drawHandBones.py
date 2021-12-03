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
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def drawhandBone(joints,order='mano',azim=0):
    assert order=='mano'
    assert len(joints.shape)==2
    fig = plt.figure(figsize=(4, 4))
    color = Constant.finger_color
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(linesg)):
        ax.plot(joints[linesg[i],0],joints[linesg[i],1],joints[linesg[i],2], marker='o',color=color[i]/255)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    set_axes_equal(ax)
    #ax.view_init(elev=-74., azim=-84)
    #plt.show()

def drawHandJointsMatplotlib(img,poseuv,order='mano',noJoints=False):
    lineidx = [0, 1, 2, 3, 17, 4, 5, 6, 18, 7, 8, 9, 20, 10, 11, 12, 19, 13, 14, 15, 16]
    preidx = [0, 0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 10, 11, 12, 0, 13, 14, 15]
    prei = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9]
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    color = Constant.finger_color
    cimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #cimg = img
    if noJoints==False:
        for i in range(len(linesg)):
            ax.plot(poseuv[linesg[i],0],poseuv[linesg[i],1], marker='o',color=color[i]/255)
    ax.imshow(cimg)
    ax.axis('off')
    #plt.show()

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


def drawHandJointsWithImgIn3DPlotly(img,uvds,texts,order='mano',outpath='my_file.html'):
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly
    import pandas as pd
    import numpy as np
    df=[]
    imagedis=400
    changdif=200
    offset=imagedis-changdif
    colors=['#F61E14','#25F614','#0B33FA','#F6FA0B']
    for idx,uvd in enumerate(uvds):
        for i in range(len(linesg)):
            df.append(go.Scatter3d(
                x=uvd[linesg[i], 0], y=uvd[linesg[i], 1], z=uvd[linesg[i], 2]-200,
                marker=dict(size=8,color=uvd[linesg[i], 2]+0,colorscale='Viridis',),
                line=dict(color=colors[idx],width=30/(idx+1))
            ))
        offset-=changdif
    #surfcolor = np.flipud()
    #print(img.shape)
    img=drawHandJoints(img,uvd)
    surfcolor = img[:,:,1]

    #cv2.imshow('surfcolor',surfcolor)
    #cv2.waitKey(0)
    x = np.linspace(0, 256, 256)
    y = np.linspace(0, 256, 256)
    X, Y = np.meshgrid(x, y)
    z = X*(1e-10)+imagedis

    annotations=[]
    offset=imagedis-changdif//3
    for text in texts:
        annotations.append(dict(
                showarrow=False,
                x=0,
                y=0,
                z=offset,
                # ax=50,
                # ay=0,
                text=text,
                # arrowhead=1,
                # xanchor="left",
                # yanchor="bottom"
            ))
        offset-=changdif

    surf = go.Surface(x=x, y=y, z=z,
                      surfacecolor=surfcolor.astype('float32')/255,
                      autocolorscale=True,
                      showscale=False
                      )
    #fig = go.Figure(data=df+[surf])
    # fig.update_layout(scene=dict(annotations=annotations), )
    fig = go.Figure(data=df)
    fig.update_layout()
    fig.write_html(outpath)
    fig.show()
    plotly.offline.plot(figure_or_data=fig, filename=outpath)

