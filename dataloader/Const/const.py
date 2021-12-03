epsilon=1e-6
MV2mano_skeidx=[0,1,2,3, 5,6,7, 13,14,15, 9,10,11, 17,18,19, 20,4,8,12,16]
RHD2mano_skeidx=[0,8,7,6, 12,11,10, 20,19,18, 16,15,14, 4,3,2,1, 5,9,13,17]
Frei2mano_skeidx = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]
Mano2frei_skeidx = [0, 13,14,15,16, 1,2,3,17,  4,5,6,18,  10,11,12,19, 7,8,9,20]
import os
manoPath='/home/csc/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl'
if not os.path.exists(manoPath):
    manoPath = '/home/shicheng/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl'
if not os.path.exists(manoPath):
    manoPath = '/home/csc/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl'

mvdatasetpaths=['/media/csc/Seagate Backup Plus Drive/dataset/7-14-1-2/mlresults/7-14-1-2_3_5result_45.pkl',
            '/media/csc/Seagate Backup Plus Drive/dataset/9-10-1-2/mlresults/9-10-1-2_1result_38.pkl',
            '/media/csc/Seagate Backup Plus Drive/dataset/9-17-1-2/mlresults/9-17-1-2_7result_45.pkl',
            '/media/csc/Seagate Backup Plus Drive/multicamera/9-25-1-2/mlresults/9-25-1-2_3result_45.pkl',
           ]
if not os.path.exists(mvdatasetpaths[0]):
    mvdatasetpaths = ['/mnt/data/shicheng/7-14-1-2/mlresults/7-14-1-2_3_5result_45.pkl',
                  '/mnt/data/shicheng/9-10-1-2/mlresults/9-10-1-2_1result_38.pkl',
                  '/mnt/data/shicheng/9-17-1-2/mlresults/9-17-1-2_7result_45.pkl',
                  '/mnt/data/shicheng/9-25-1-2/mlresults/9-25-1-2_3result_45.pkl',
                  ]
if not os.path.exists(mvdatasetpaths[0]):
    mvdatasetpaths = ['/mnt/ssd/shicheng/7-14-1-2/mlresults/7-14-1-2_3_5result_45.pkl',
                  '/mnt/ssd/shicheng/9-10-1-2/mlresults/9-10-1-2_1result_38.pkl',
                  '/mnt/ssd/shicheng/9-17-1-2/mlresults/9-17-1-2_7result_45.pkl',
                  '/mnt/ssd/shicheng/9-25-1-2/mlresults/9-25-1-2_3result_45.pkl',
                  ]
if not os.path.exists(mvdatasetpaths[0]):
    mvdatasetpaths = ['/mnt/ssd/csc/7-14-1-2/mlresults/7-14-1-2_3_5result_45.pkl',
                  '/mnt/ssd/csc/9-10-1-2/mlresults/9-10-1-2_1result_38.pkl',
                  '/mnt/ssd/csc/9-17-1-2/mlresults/9-17-1-2_7result_45.pkl',
                  '/mnt/ssd/csc/9-25-1-2/mlresults/9-25-1-2_3result_45.pkl',
                  ]