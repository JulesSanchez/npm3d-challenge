import open3d as o3d 
import numpy as np 
from copy import copy
from plyfile import PlyData, PlyElement

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.subsampler import get_even_number
from utils.features_computation import compute_covariance_features

NAMEFILES = ['MiniLille1','MiniLille2','MiniParis1']
NAMETEST = ['MiniDijon9']

def load_point_cloud(name,down_sample=False):
    """Load the point cloud.
    
    Parameters
    ----------
    name : str
        filename of the point cloud to load.
    
    Returns
    -------
    points : ndarray
        Point cloud data (N_points, 3).
    ndarray
        Labels of the points in the cloud (N_points,).
        0 is not a "real" label, it corresponds to unclassified data points.
    pcd_tree : KDTree
        Tree to perform neighbor search in the point cloud (to compute features).
    """
    plydata = PlyData.read(name)
    pcd = o3d.io.read_point_cloud(name)
    if down_sample:
        downpcd = pcd.voxel_down_sample(voxel_size=down_sample)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    return np.asarray(pcd.points), np.asarray(plydata.elements[0].data['class']), pcd_tree

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


RADIUS = 0.5


class MyPointCloud(Dataset):
    """Basic point cloud dataset."""
    def __init__(self, filepath: str):
        super().__init__()
        points, labels, tree = load_point_cloud(filepath)
        self.points = points
        self.labels = labels
        self.tree = tree
    
    def __len__(self):
        return len(self.points)
    
    
    def __getitem__(self, idx: int):
        """Use `~MyCloudSampler` as a sampler inside of your
        DataLoader instance instead."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        labels = self.labels[idx]
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        return torch.from_numpy(self.points[idx]), labels
     
    
    def get_sample(self, size=1000):
        subcloud, sublabels = get_even_number(
            self.points, self.labels, size)

        features = compute_covariance_features(
            subcloud, self.points, self.tree, radius=RADIUS)
        
        subcloud = torch.from_numpy(subcloud).float()
        sublabels = torch.from_numpy(sublabels).long()
        features = [torch.from_numpy(f).unsqueeze(1).float() for f in features]
        
        return subcloud, sublabels, features
