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

def preprocess(name,path_output=False,voxel_size = 0.5, labels = True):
    #path_output should not have an extension
    #name need an extension
    plydata = PlyData.read(name)
    pcd = o3d.io.read_point_cloud(name)
    if labels :
        labels = np.asarray(plydata.elements[0].data['class'])
        indices = labels > 0
        labels = labels[indices].astype(np.float32)
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[indices])
        max_label = np.max(labels)
        labels /= max_label
        pcd.colors = o3d.utility.Vector3dVector(np.ones(shape=(labels.shape[0],3))*labels[:,None])
        downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        downpcd.colors = o3d.utility.Vector3dVector(np.round(downpcd.colors*max_label))
        labels = np.asarray(downpcd.colors)[:,0]
        print('Number of points inside the downsampled point cloud : ' + str(len(labels)))
        if path_output:
            o3d.io.write_point_cloud(path_output+'.ply',downpcd)
            np.savetxt(path_output+'.txt',labels.astype(int),delimiter='\n',fmt='%i')
        return np.asarray(downpcd.points), np.asarray(downpcd.colors)
    else:
        downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        if path_output:
            o3d.io.write_point_cloud(path_output+'.ply',downpcd)
        return np.asarray(downpcd.points), None

def load_downsampled_point_cloud(path_output):
    #path_output should not have an extension
    point_cloud, tree = load_point_cloud(path_output+'.ply')
    try:
        labels = np.loadtxt(path_output+'.txt')
        return point_cloud, labels, tree
    except:
        return point_cloud, tree


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
    try:
        return np.asarray(pcd.points), np.asarray(plydata.elements[0].data['class']), pcd_tree
    except:
        return np.asarray(pcd.points), pcd_tree


def cross_val():
    """
    Separate the data between training, val, and test point clouds.
    
    Returns
    -------
    folds : dict
        Keys: training: paths for the training files, val: paths to the validation files,
        test: path to the test file
    """
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


def write_results(path, labels,test=True):
    if test:
        np.savetxt(path+NAMETEST[0]+'nodes.txt',labels.astype(int),fmt='%i')
    else:
        np.savetxt(path+'nodes.txt',np.array(labels).astype(int),fmt='%i')
    return 



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


class VoxelizedPointCloud(Dataset):
    """Use as input for PointNet."""
    def __init__(self, filepath: str):
        receive = load_downsampled_point_cloud(filepath)
        if len(receive) == 3:
            self.points, self.labels, self._tree = receive
        else:
            self.points, self._tree = receive
            self.labels = None
        
    def __getitem__(self, idx: int):
        poi = torch.from_numpy(self.points[idx]).float()
        if self.labels is not None:
            lab = self.labels[idx]
            if isinstance(lab, np.ndarray):
                lab = torch.from_numpy(lab).long()
            return poi, lab
        return poi

    def get(self):
        if self.labels is not None:
            return torch.from_numpy(self.points).float(), torch.from_numpy(self.labels).long()
        else:
            return torch.from_numpy(self.points).float()


class ConcatVoxelizedPointCloud(torch.utils.data.ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

    @property
    def proportions(self):
        return np.array([len(d) for d in self.datasets]) / len(self)

    def get(self, size=1000):
        dataset_idx = np.random.choice(len(self.datasets), p=self.proportions)
        d: VoxelizedPointCloud = self.datasets[dataset_idx]
        return d.get()


