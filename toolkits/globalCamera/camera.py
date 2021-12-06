from typing import NamedTuple
import numpy as np


def perspective_projection(xyz_point, camera):
    if xyz_point.ndim == 1:
        uvd_point = np.zeros((3))
        uvd_point[0] = xyz_point[0] * camera.fx / (xyz_point[2]) + camera.cx
        uvd_point[1] = xyz_point[1] * camera.fy / (xyz_point[2]) + camera.cy
        uvd_point[2] = xyz_point[2]
    elif xyz_point.ndim == 2:
        num_point = xyz_point.shape[0]
        uvd_point = np.zeros((num_point, 3))
        uvd_point[:, 0] = xyz_point[:, 0] * \
            camera.fx / xyz_point[:, 2] + camera.cx
        uvd_point[:, 1] = xyz_point[:, 1] * \
            camera.fy / xyz_point[:, 2] + camera.cy
        uvd_point[:, 2] = xyz_point[:, 2]
    else:
        raise ValueError('unknown input point shape')

    return uvd_point

def perspective_back_projection(uvd_point, camera):
    if uvd_point.ndim == 1:
        xyz_point = np.zeros((3))
        xyz_point[0] = (uvd_point[0] - camera.cx) * uvd_point[2] / camera.fx
        xyz_point[1] = (uvd_point[1] - camera.cy) * uvd_point[2] / camera.fy
        xyz_point[2] = uvd_point[2]
    elif uvd_point.ndim == 2:
        num_point = uvd_point.shape[0]
        xyz_point = np.zeros((num_point, 3))
        xyz_point[:, 0] = (uvd_point[:, 0] - camera.cx) * \
            uvd_point[:, 2] / camera.fx
        xyz_point[:, 1] = (uvd_point[:, 1] - camera.cy) * \
            uvd_point[:, 2] / camera.fy
        xyz_point[:, 2] = uvd_point[:, 2]
    else:
        raise ValueError('unknown input point shape')
    return xyz_point


class RS3_434(NamedTuple):
    fx: float = 617.618
    fy: float = 617.618
    cx: float = 311.322
    cy: float = 236.829
    depth_unit: float = 0.124987

class RS4_665(NamedTuple):
    fx: float = 616.754
    fy: float = 615.578
    cx: float = 313.754
    cy: float = 241.617
    depth_unit: float = 1.0

class RS4_792(NamedTuple):
    fx: float = 614.504
    fy: float = 614.177
    cx: float = 313.787
    cy: float = 230.687
    depth_unit: float = 1.0

class RS4_866(NamedTuple):
    fx: float = 609.516
    fy: float = 609.689
    cx: float = 312.865
    cy: float = 233.208
    depth_unit: float = 1.0


class RS4_035(NamedTuple):
    fx: float = 619.598
    fy: float = 619.116
    cx: float = 325.345
    cy: float = 245.441
    depth_unit: float = 1.0
class RS4_037(NamedTuple):
    fx: float = 615.85
    fy: float = 615.477
    cx: float = 316.062
    cy: float = 247.156
    #intrinsics depth width: 640, height: 480, ppx: 323.303, ppy: 247.025, fx: 598.493, fy: 598.493, model: 4, coeffs: [0, 0, 0, 0, 0]

    depth_unit: float = 1.0
class RS4_038(NamedTuple):
    fx: float = 619.475
    fy: float = 619.189
    cx: float = 313.715
    cy: float = 223.921
    depth_unit: float = 1.0
class RS4_076(NamedTuple):
    fx: float = 615.665
    fy: float = 615.09
    cx: float = 306.514
    cy: float = 240.344
    depth_unit: float = 1.0

class RS4_03510(NamedTuple):
    cx: float = 648.018
    cy: float = 368.161
    fx: float = 929.397
    fy: float = 928.673
    depth_unit: float = 1.0
class RS4_03710(NamedTuple):
    cx: float = 634.093
    cy: float = 370.733
    fx: float = 923.775
    fy: float = 923.215
    depth_unit: float = 1.0
class RS4_03810(NamedTuple):
    cx: float = 630.572
    cy: float = 335.881
    fx: float = 929.213
    fy: float = 928.783
    depth_unit: float = 1.0
class RS4_07610(NamedTuple):
    cx: float = 619.77
    cy: float = 360.516
    fx: float = 923.497
    fy: float = 922.635
    depth_unit: float = 1.0

class RS4_361(NamedTuple):
    fx: float = 617.798
    fy: float = 617.741
    cx: float = 307.941
    cy: float = 228.490
    depth_unit: float = 1.0


CameraSeries = [
    '746112061866',
    '746112060792',
    '614205001434',
    '839112061665',
    '823112060361',
    '840412062037',
    '840412062038',
    '840412062076',
    '840412062035',
    '84041206203510',
    '84041206203710',
    '84041206203810',
    '84041206207610',
]

CameraIntrinsics = {
    '614205001434': RS3_434(),
    '839112061665': RS4_665(),
    '746112060792': RS4_792(),
    '746112061866': RS4_866(),
    '840412062035': RS4_035(),
    '840412062037': RS4_037(),
    '840412062038': RS4_038(),
    '840412062076': RS4_076(),
    '84041206203510': RS4_03510(),
    '84041206203710': RS4_03710(),
    '84041206203810': RS4_03810(),
    '84041206207610': RS4_07610(),
    '823112060361': RS4_361()
}