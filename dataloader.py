from toolkits.toolkit import MultiviewDatasetDemo
from torch.utils.data import Dataset

class FreiDateset3D(Dataset):
    def __init__(self):
        path = "/media/csc/Seagate Backup Plus Drive/dataset/7-14-1-2"
        manoPath = '/home/csc/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl'
        self.demo=MultiviewDatasetDemo(loadManoParam=True,file_path=path,manoPath=manoPath)
    def __len__(self):
        return self.demo.joints.shape[0]

    def __getitem__(self, idx):
        v=idx%4
        idx=idx//4
        depth=self.demo.getDepth(idx,uselist=True)[v]
        rgb=self.demo.getImgs(idx,uselist=True)[v]
        kp=self.demo.getPose3D(idx,v)
        return depth,rgb,kp
