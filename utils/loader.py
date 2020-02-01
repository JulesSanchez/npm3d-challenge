import open3d as o3d 
import numpy as np 
from copy import copy
from plyfile import PlyData, PlyElement

NAMEFILES = ['MiniLille1','MiniLille2','MiniParis1']
NAMETEST = ['MiniDijon9']

def load_point_cloud(name,down_sample=False):
    """Load the point cloud.
    
    Parameters
    ----------
    name : str
        filename of the point cloud to load
    
    Returns
    -------
    ndarray
        Point cloud data (N_points, 3)
    ndarray
        labels of the points in the cloud (N_points,)
    pcd_tree : KDTree
        KDTree to perform neighbor search in the point cloud (to compute features)
    """
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
        dic['val'] = [NAMEFILES[k]]
        copy_N = copy(NAMEFILES)
        copy_N.remove(NAMEFILES[k])
        dic['training'] = copy_N
        dic['test'] = NAMETEST
        folds.append(dic)
    return folds
