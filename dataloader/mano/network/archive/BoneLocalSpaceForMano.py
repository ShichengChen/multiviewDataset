from cscPy.mano.network.manolayer import MANO_SMPL
from cscPy.mano.network.utils import *
from cscPy.mano.network.utilsSmallFunctions import *
from cscPy.Const.const import *
mano_right = MANO_SMPL(manoPath, ncomps=45,oriorder=True,cuda='cpu')
F=mano_right.faces
##norm of palm is z direction
##wirst to mmcp is y direction
#n1:plane wrist->tmcp->imcp
#n2:plane wrist->imcp->mmcp
#n3:plane wrist->mmcp->rmcp
#n4:plane wrist->rmcp->pmcp
wristp=mano_right.J[0:1]
tempJ=getBatch(mano_right.J-wristp)
tempV=getBatch(mano_right.v_template.cpu().numpy()-wristp)
n1=torch.cross(tempJ[:, 13] - tempJ[:, 0], tempJ[:, 1] - tempJ[:, 13], dim=1)
n2=torch.cross(tempJ[:, 1] - tempJ[:, 0], tempJ[:, 4] - tempJ[:, 1], dim=1)
n3=torch.cross(tempJ[:, 4] - tempJ[:, 0], tempJ[:, 10] - tempJ[:, 4], dim=1)
n4=torch.cross(tempJ[:, 10] - tempJ[:, 0], tempJ[:, 7] - tempJ[:, 10], dim=1)
norms=[n2,n2,n2,
       (n3+n2)/2,(n3+n2)/2,(n3+n2)/2,
       (n3+n4)/2,(n3+n4)/2,(n3+n4)/2,
       n4,n4,n4,
       n1,n1,n1,]
ytempJs=[
tempJ[:,2]-tempJ[:,1],tempJ[:,3]-tempJ[:,2],tempJ[:,17]-tempJ[:,3],
tempJ[:,5]-tempJ[:,4],tempJ[:,6]-tempJ[:,5],tempJ[:,18]-tempJ[:,6],
tempJ[:,8]-tempJ[:,7],tempJ[:,9]-tempJ[:,8],tempJ[:,20]-tempJ[:,9],
tempJ[:,11]-tempJ[:,10],tempJ[:,12]-tempJ[:,11],tempJ[:,19]-tempJ[:,12],
tempJ[:,14]-tempJ[:,13],tempJ[:,15]-tempJ[:,14],tempJ[:,16]-tempJ[:,15],
]

bonespace=[
    np.eye(4),
]
zaxis=getBatch(torch.tensor([0,0,1]))
yaxis=getBatch(torch.tensor([0,1,0]))

for bonei in range(15):

    alignr=planeAlignment(norms[bonei],ytempJs[bonei],zaxis,yaxis)

    indexbone0J = (alignr.reshape(1, 3, 3)) @ tempJ.reshape(1, 21, 3, 1)
    rt = get32fTensor(comebineRt3D(alignr.reshape(3, 3), -indexbone0J[:, bonei + 1:bonei + 2, :, :].reshape(-1)))
    #indexbone0V = (alignr.reshape(1, 3, 3)) @ (tempV.reshape(1, 778, 3, 1)) - indexbone0J[:, bonei+1:bonei+2, :, :]
    homov=getHomo3D(tempV.reshape(1, 778, 3, 1)).reshape(1,778,4,1)
    indexbone0V = rt @ homov
    with open('/home/csc/objs/bone'+str(bonei)+'.obj', 'w') as the_file:
        for i in range(778):
            the_file.write('v '+str(float(indexbone0V[0,i,0,0]))+" "+str(float(indexbone0V[0,i,1,0]))
                           +" "+str(float(indexbone0V[0,i,2,0]))+" 1.0\n")
        for i in range(F.shape[0]):
            the_file.write('f '+str(int(F[i,0]+1))+" "+str(int(F[i,1]+1))+" "+str(int(F[i,2]+1))+"\n")
    bonespace.append(rt.numpy())

print(bonespace)
