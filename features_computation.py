
import numpy as np
# Import functions from scikit-learn
from sklearn.neighbors import KDTree
from physt import h1, h2, histogramdd
import time 
EPSILON = 1e-10 

def tetrahedron_calc_volume(a, b, c, d):
    test = np.abs(np.linalg.det(np.stack(((a-b).T , (b-c).T, (c-d).T)).T))/6
    return test

def area(a, b, c) :
    return 0.5 * np.linalg.norm( np.cross( b-a, c-a ), axis=1 )

def get_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.diag(np.inner(ba, bc)) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle

def adaptative_bin(samples,bins):
    total_number = len(samples)
    bin_s = [np.percentile(samples,i*10) for i in range(11)]
    return np.array(bin_s)


def local_PCA(points):

    barycenter = np.mean(points,axis=0)
    centered_points = points - barycenter
    M_cov = np.dot(centered_points.T,centered_points)/len(centered_points)
    eigenvalues, eigenvectors = np.linalg.eigh(M_cov)
    return eigenvalues, eigenvectors

def neighborhood_PCA(query_points, cloud_points, tree, radius):

    all_eigenvalues = np.zeros((query_points.shape[0], 3)) 
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))
    for i in range(query_points.shape[0]):
        point = query_points[i]
        _, indices, _ = tree.search_radius_vector_3d(point, radius)
        neighbors = cloud_points[indices[1:],:]
        if len(neighbors)==0:
            neighbors = point.reshape(1,3)
        local_eigenvalues, local_eigenvectors = local_PCA(neighbors)
        all_eigenvalues[i] = local_eigenvalues
        all_eigenvectors[i] = local_eigenvectors
    return all_eigenvalues, all_eigenvectors


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

    for i in range(query_points.shape[0]):
        local_D4 = np.zeros(pulls) 
        local_A3 = np.zeros(pulls)
        point = query_points[i]
        _, indices, _ = tree.search_radius_vector_3d(point, radius)
        neighbors = cloud_points[indices[1:],:]
        if len(neighbors)< 4*pulls:
            _, indices, _ = tree.search_knn_vector_3d(point, knn=4*pulls+1)
            neighbors = cloud_points[indices[1:],:]
        length = np.arange(len(neighbors))

        indicesD1 = np.random.choice(length,size=pulls,replace=False)
        indicesD2 = np.random.choice(length,size=(2,pulls),replace=False)
        indicesD3 = np.random.choice(length,size=(3,pulls),replace=False)
        indicesD4 = np.random.choice(length,size=(4,pulls),replace=False)
        indicesA3 = np.random.choice(length,size=(3,pulls),replace=False)
        centroid = np.mean(neighbors,axis=0)
        D1[i] = np.linalg.norm(neighbors[indicesD1]-centroid,axis=1)
        D2[i] = np.linalg.norm(neighbors[indicesD2[0,:]]-neighbors[indicesD2[1,:]],axis=1)
        D3[i] = area(neighbors[indicesD3[0,:]],neighbors[indicesD3[1,:]],neighbors[indicesD3[2,:]])
        D4[i] = tetrahedron_calc_volume(neighbors[indicesD4[0]],neighbors[indicesD4[1]],neighbors[indicesD4[2]],neighbors[indicesD4[3]])
        A3[i] = get_angle(neighbors[indicesA3[0]],neighbors[indicesA3[1]],neighbors[indicesA3[2]])

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

