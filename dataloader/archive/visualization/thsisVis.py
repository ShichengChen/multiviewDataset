
import matplotlib.pyplot as plt
print("1")
import numpy as np

print("2")
joint_color = [(255, 0, 0)] * 1 + \
              [(25, 255, 25)] * 4 + \
              [(212, 0, 255)] * 4 + \
              [(0, 230, 230)] * 4 + \
              [(179, 179, 0)] * 4 + \
              [(0, 0, 255)] * 4
linecolor=np.array([[0, 0, 255],[25, 255, 25],[212, 0, 255],[0, 230, 230],[179, 179, 0]])
#linecolor=['green','magenta','yellow','cyan','white']
linesg = [[0, 1, 2, 3, 4], [0, 5, 6, 7,8], [0, 9, 10,11,12], [0, 13, 14,15,16], [0, 17,18,19,20]]
wristcolor = (255, 0, 0)
indexcolor = (25, 255, 25)
middlecolor = (212, 0, 255)
ringcolor = (0, 230, 230)
pinkycolor = (179, 179, 0)
thumbcolor = (255, 153, 153)
joint_color = [wristcolor] * 1 + \
              [indexcolor] * 4 + \
              [middlecolor] * 4 + \
              [ringcolor] * 4 + \
              [pinkycolor] * 4 + \
              [thumbcolor] * 4
mano_joints_color = np.array([wristcolor] + [indexcolor] * 3 + [middlecolor] * 3 +
                             [pinkycolor] * 3 + [ringcolor] * 3 + [thumbcolor] * 4 + \
                             [indexcolor, middlecolor, ringcolor, pinkycolor])
finger_color = np.array([indexcolor] + [middlecolor] + [ringcolor] + [pinkycolor] + [thumbcolor])

x=[0,5,10,12,12, 9,9,9,9,       2,2,2,2,      -4,-4,-4,-4,  -9,-10,-10,-10]
v=[0,5,10,12,12, 7,11,7,11,       2,2,2,2,      -4,-4,-4,-4,  -9,-10,-10,-10]
y=[0,2,9,13,18,  17,24,24,24,   18,25,31,37,  17,24,30,35,  16,23,28,33]
z=[0,0,0,0 ,0 ,  0, 0, 6, 12,    0,0,0,0,      0,0,0,0,      0,0,0,0]
x=np.array(x)
v=np.array(v)
y=np.array(y)
z=np.array(z)
if __name__ == "__main__":
    def fun(a,b):return a*0+b*0.00001+9
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Y,Z = np.meshgrid(np.arange(-40, 40, 1), np.arange(-40, 40, 1))
    X = np.zeros_like(Y)+9
    ax.plot_surface(X, Y, Z, alpha=0.5)
    #ax.plot_surface(np.array([9,9,9,9]), [100,100,-100,-100], [100,-100,100,-100])

    color = finger_color
    for i in range(len(linesg)):
        if(i==1):
            ax.plot(v[linesg[i]], y[linesg[i]], z[linesg[i]], marker='o',linestyle='dashed', color=color[i] / 255)
            #ax.plot(x[linesg[i]], y[linesg[i]], z[linesg[i]], marker='o', color=color[i] / 255)
        else:
            ax.plot(x[linesg[i]], y[linesg[i]], z[linesg[i]], marker='o', color=color[i] / 255)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    ax.set_xlim3d(-25, 25)
    ax.set_ylim3d(-10, 40)
    ax.set_zlim3d(-5, 35)

    plt.show()
