
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree


EPSILON = 1e-10 

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