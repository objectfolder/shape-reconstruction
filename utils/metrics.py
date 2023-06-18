import numpy as np
from scipy.spatial import cKDTree as KDTree


def TrimeshChamferDistance(gt_points, est_points):
    if len(gt_points.shape) == 2:
        # one direction
        gen_points_kd_tree = KDTree(est_points)
        one_distances, _ = gen_points_kd_tree.query(gt_points)
        gt_to_gen_chamfer = np.mean(np.abs(one_distances))

        # other direction
        gt_points_kd_tree = KDTree(gt_points)
        two_distances, _ = gt_points_kd_tree.query(est_points)
        gen_to_gt_chamfer = np.mean(np.abs(two_distances))

        return 0.5*gt_to_gen_chamfer + 0.5*gen_to_gt_chamfer
    elif len(gt_points.shape) == 3:  # batch
        chamfer = []
        for i in range(gt_points.shape[0]):
            # one direction
            gen_points_kd_tree = KDTree(est_points[i])
            one_distances, _ = gen_points_kd_tree.query(gt_points[i])
            gt_to_gen_chamfer = np.mean(np.abs(one_distances))

            # other direction
            gt_points_kd_tree = KDTree(gt_points[i])
            two_distances, _ = gt_points_kd_tree.query(est_points[i])
            gen_to_gt_chamfer = np.mean(np.abs(two_distances))

            chamfer.append(0.5*gt_to_gen_chamfer + 0.5*gen_to_gt_chamfer)
        return np.mean(chamfer)
