
import time
import numpy as np


def backproject(depth, intrinsics, instance_mask):
    """ Back-projection, use opencv camera coordinate frame.

    """
    cam_fx = intrinsics[0, 0]
    cam_fy = intrinsics[1, 1]
    cam_cx = intrinsics[0, 2]
    cam_cy = intrinsics[1, 2]

    non_zero_mask = (depth > 0)
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)
    idxs = np.where(final_instance_mask)

    z = depth[idxs[0], idxs[1]]
    x = (idxs[1] - cam_cx) * z / cam_fx
    y = (idxs[0] - cam_cy) * z / cam_fy
    pts = np.stack((x, y, z), axis=1)

    return pts, idxs

