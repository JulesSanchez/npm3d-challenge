import numpy as np
import time 
from utils.features.pca import *

EPSILON = 1e-10


def compute_covariance_features(query_points, cloud_points, tree, radius):

    all_eigenvalues, all_eigenvectors = neighborhood_PCA(query_points,cloud_points,tree,radius)
    normals = all_eigenvectors[:,:,0].reshape(-1,3)
    ez = np.array([0,0,1])
    verticality = 2*np.arcsin(np.abs(np.dot(normals,ez)))/np.pi
    linearity = 1 - all_eigenvalues[:,1]/(all_eigenvalues[:,2]+ EPSILON)
    planarity = (all_eigenvalues[:,1]-all_eigenvalues[:,0])/(all_eigenvalues[:,2]+ EPSILON)
    sphericity = all_eigenvalues[:,0]/(all_eigenvalues[:,2]+ EPSILON)
    omnivariance = all_eigenvalues[:,0]*all_eigenvalues[:,1]*all_eigenvalues[:,2]
    omnivariance = np.sign(omnivariance) * (np.abs(omnivariance)) ** (1 / 3)
    anisotropy = (all_eigenvalues[:,2]-all_eigenvalues[:,0])/(all_eigenvalues[:,2]+ EPSILON)    
    eigenentropy = -(np.sum(all_eigenvalues*np.log(all_eigenvalues+EPSILON),axis=1))
    sumeigen = np.sum(all_eigenvalues,axis=1)
    change_curvature = all_eigenvalues[:,0]/(sumeigen+EPSILON)
    return verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature

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

