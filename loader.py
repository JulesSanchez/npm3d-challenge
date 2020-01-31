import open3d as o3d 
from features_computation import compute_covariance_features
import numpy as np 
from copy import copy
from plyfile import PlyData, PlyElement
NAMEFILES = ['MiniLille1','MiniLille2','MiniParis1']

def load_point_cloud(name,down_sample=False):
    plydata = PlyData.read(name)
    pcd = o3d.io.read_point_cloud(name)
    if down_sample:
        downpcd = pcd.voxel_down_sample(voxel_size=down_sample)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    return np.asarray(pcd.points), plydata.elements[0].data['class'], pcd_tree

def cross_val():
    folds = []
    for k in range(len(NAMEFILES)):
        dic = {}
        dic['test'] = [NAMEFILES[k]]
        copy_N = copy(NAMEFILES)
        copy_N.remove(NAMEFILES[k])
        dic['training'] = copy_N
        folds.append(dic)
    return folds
