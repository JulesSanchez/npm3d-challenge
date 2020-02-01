import numpy as np
import time 
from utils.features.pca import *

EPSILON = 1e-10


def shape_distributions(query_points, cloud_points, tree, radius=2.5, pulls=255, bins=10):
    #Pulls is the number of time we compute shape distributions for each points

    D1 = np.zeros((query_points.shape[0],255))
    D2 = np.zeros((query_points.shape[0],255))
    D3 = np.zeros((query_points.shape[0],255))
    D4 = np.zeros((query_points.shape[0],255)) 
    A3 = np.zeros((query_points.shape[0],255))
    
    shape_dist_inner_loop(query_points, cloud_points, tree,
                          D1, D2, D3, D4, A3,
                          radius, pulls, bins)
    
    if isinstance(bins, int):
        old_bins = bins
        bins = []
        bins.append(adaptative_bin(D1.flatten(),old_bins))
        bins.append(adaptative_bin(D2.flatten(),old_bins))
        bins.append(adaptative_bin(D3.flatten(),old_bins))
        bins.append(adaptative_bin(D4.flatten(),old_bins))
        bins.append(adaptative_bin(A3.flatten(),old_bins))
        D1 = np.array([np.histogram(D1[i],bins=bins[0])[0] for i in range(len(D1))]).T
        D2 = np.array([np.histogram(D2[i],bins=bins[1])[0] for i in range(len(D2))]).T
        D3 = np.array([np.histogram(D3[i],bins=bins[2])[0] for i in range(len(D3))]).T
        D4 = np.array([np.histogram(D4[i],bins=bins[3])[0] for i in range(len(D4))]).T
        A3 = np.array([np.histogram(A3[i],bins=bins[4])[0] for i in range(len(A3))]).T
    else:
        D1 = np.array([np.histogram(D1[i],bins=bins[0])[0] for i in range(len(D1))]).T
        D2 = np.array([np.histogram(D2[i],bins=bins[1])[0] for i in range(len(D2))]).T
        D3 = np.array([np.histogram(D3[i],bins=bins[2])[0] for i in range(len(D3))]).T
        D4 = np.array([np.histogram(D4[i],bins=bins[3])[0] for i in range(len(D4))]).T
        A3 = np.array([np.histogram(A3[i],bins=bins[4])[0] for i in range(len(A3))]).T

    return D1, D2, D3, D4, A3, bins

