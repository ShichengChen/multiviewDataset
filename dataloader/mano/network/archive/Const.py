import numpy as np
import torch
# joint mapping indices from mano to bighand
mano2bighand_skeidx = [0, 13, 1, 4, 10, 7, 14, 15, 16, 2, 3, 17, 5, 6, 18, 11, 12, 19, 8, 9, 20]
STB2Bighand_skeidx = [0, 17, 13, 9, 5, 1, 18, 19, 20, 14, 15, 16, 10, 11, 12, 6, 7, 8, 2, 3, 4]
Bighand2mano_skeidx = [0, 2, 9, 10, 3, 12, 13, 5, 18, 19, 4, 15, 16, 1, 6, 7, 8, 11, 14, 17, 20]
RHD2Bighand_skeidx = [0,4,8,12,16,20,3,2,1,7,6,5,11,10,9,15,14,13,19,18,17]
SynthHands2Bighand_skeidx=[0,1,5,9,13,17,2,6,10,14,18,3,7,11,15,19,4,8,12,16,20]
MV2mano_skeidx=[0,1,2,3, 5,6,7, 13,14,15, 9,10,11, 17,18,19, 20,4,8,12,16]




