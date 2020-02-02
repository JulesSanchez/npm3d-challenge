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

CLASSES = ['Unclassified', 'Ground', 'Building',
           'Poles', 'Pedestrians', 'Cars', 'Vegetation']


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




class MyPointCloud(Dataset):
    """Basic point cloud Dataset structure.
    
    Includes a method to subsample and compute features on-the-fly."""
    def __init__(self, filepath: str, radius: float = 0.5, multiscale=None):
        super().__init__()
        points, labels, tree = load_point_cloud(filepath)
        self.points = points
        self.labels = labels
        self.tree = tree
        self.radius = radius
        self.multiscale = multiscale
    
    def __len__(self):
        return len(self.points)
    
    
    def __getitem__(self, idx: int):
        """Allows to index into the point cloud.
        
        Use the `get_sample` method inside of your training loop instead instead."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        labels = self.labels[idx]
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        return torch.from_numpy(self.points[idx]), labels
     
    
    def get_sample(self, size=1000):
        subcloud, sublabels = get_even_number(
            self.points, self.labels, size)

        # loop over the multi-scale radii
        if self.multiscale is not None:
            features = []
            for r in self.multiscale:
                features += compute_covariance_features(
                    subcloud, self.points, self.tree, radius=r)
        else:
            # use radius param
            features = compute_covariance_features(
                subcloud, self.points, self.tree, radius=self.radius)

        
        subcloud = torch.from_numpy(subcloud).float()
        sublabels = torch.from_numpy(sublabels).long()
        features = [torch.from_numpy(f).unsqueeze(1).float() for f in features]
        
        return subcloud, sublabels, features


class ConcatPointClouds(torch.utils.data.ConcatDataset):
    """Fuse together multiple `~MyPointCloud` point cloud datasets.
    
    The `get_sample` method samples randomly from either one or the other datasets."""
    def __init__(self, datasets):
        super().__init__(datasets)
    
    @property
    def proportions(self):
        return np.array([len(d) for d in self.datasets]) / len(self)
    
    def get_sample(self, size=1000):
        dataset_idx = np.random.choice(len(self.datasets), p=self.proportions)
        return self.datasets[dataset_idx].get_sample(size)
