# cython: infer_types=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
import numpy as np
cimport numpy as np
from numpy import ndarray
from numpy cimport ndarray

cdef double EPSILON = 1e-10


cpdef ndarray[double] tetrahedron_calc_volume(ndarray a, ndarray b, ndarray c, ndarray d):
    return np.abs(np.linalg.det(np.stack(((a-b).T , (b-c).T, (c-d).T)).T))/6

cpdef ndarray[double] area(ndarray a, ndarray b, ndarray c):
    return 0.5 * np.linalg.norm( np.cross( b-a, c-a ), axis=1)

cpdef ndarray[double] get_angle(ndarray a, ndarray b, ndarray c):
    cdef ndarray ba = a - b
    cdef ndarray bc = c - b
    cdef ndarray cosine_angle = np.diag(np.inner(ba, bc)) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cdef ndarray angle = np.arccos(cosine_angle)
    return angle

cpdef ndarray[double] adaptative_bin(ndarray samples, int bins):
    cdef ndarray[double] bin_s = np.empty(bins+1)
    for i in range(bins+1):
        bin_s[i] = np.percentile(samples,i*100./bins)
    return bin_s


cpdef local_PCA(ndarray points):
    cdef ndarray[double] barycenter = np.mean(points,axis=0)
    cdef ndarray centered_points = points - barycenter
    cdef ndarray M_cov = np.dot(centered_points.T,centered_points)/len(centered_points)
    cdef ndarray eigenvalues, eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(M_cov)
    return eigenvalues, eigenvectors


cpdef neighborhood_PCA(ndarray query_points, ndarray cloud_points, tree, double radius):
    cdef ndarray all_eigenvalues = np.empty((query_points.shape[0], 3)) 
    cdef ndarray all_eigenvectors = np.empty((query_points.shape[0], 3, 3))
    cdef ndarray[double] point
    cdef ndarray[double] local_eigenvalues
    cdef ndarray[double, ndim=2] local_eigenvectors
    cdef ndarray[double, ndim=2] neighbors
    cdef Py_ssize_t i
    cdef Py_ssize_t n_points = query_points.shape[0] 
    for i in range(n_points):
        point = query_points[i]
        _, indices, _ = tree.search_radius_vector_3d(point, radius)
        neighbors = cloud_points[indices[1:]]
        if len(neighbors)==0:
            neighbors = point.reshape(1,3)
        local_eigenvalues, local_eigenvectors = local_PCA(neighbors)
        all_eigenvalues[i] = local_eigenvalues
        all_eigenvectors[i] = local_eigenvectors
    return all_eigenvalues, all_eigenvectors

def compute_covariance_features(ndarray query_points, ndarray cloud_points, tree, double radius):
    cdef ndarray all_eigenvalues, all_eigenvectors
    all_eigenvalues, all_eigenvectors = neighborhood_PCA(query_points,cloud_points,tree,radius)
    cdef ndarray normals = all_eigenvectors[:,:,0].reshape(-1,3)
    cdef ndarray ez = np.array([0,0,1])
    cdef ndarray verticality = 2*np.arcsin(np.abs(np.dot(normals,ez)))/np.pi
    cdef ndarray linearity = 1 - all_eigenvalues[:,1]/(all_eigenvalues[:,2]+ EPSILON)
    cdef ndarray planarity = (all_eigenvalues[:,1]-all_eigenvalues[:,0])/(all_eigenvalues[:,2]+ EPSILON)
    cdef ndarray sphericity = all_eigenvalues[:,0]/(all_eigenvalues[:,2]+ EPSILON)
    cdef ndarray omnivariance = all_eigenvalues[:,0]*all_eigenvalues[:,1]*all_eigenvalues[:,2]
    omnivariance = np.sign(omnivariance) * (np.abs(omnivariance)) ** (1 / 3)
    cdef ndarray anisotropy = (all_eigenvalues[:,2]-all_eigenvalues[:,0])/(all_eigenvalues[:,2]+ EPSILON)    
    cdef ndarray eigenentropy = -(np.sum(all_eigenvalues*np.log(all_eigenvalues+EPSILON),axis=1))
    cdef ndarray sumeigen = np.sum(all_eigenvalues,axis=1)
    cdef ndarray change_curvature = all_eigenvalues[:,0]/(sumeigen+EPSILON)
    return verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature


cpdef shape_dist_inner_loop(ndarray query_points, ndarray cloud_points, tree,
    ndarray D1, ndarray D2, ndarray D3, ndarray D4, ndarray A3,
    double radius, int pulls, bins):
    cdef Py_ssize_t i
    cdef Py_ssize_t n_points = query_points.shape[0]
    cdef ndarray[double, ndim=1] point
    # cdef ndarray local_D4, local_A3
    cdef ndarray[double, ndim=2] neighbors
    cdef ndarray[long, ndim=1] indicesD1
    cdef ndarray[long, ndim=2] indicesD2, indicesD3, indicesD4, indicesA3
    cdef ndarray[double] centroid
    cdef ndarray length
    for i in range(n_points):
        # local_D4 = np.zeros(pulls) 
        # local_A3 = np.zeros(pulls)
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

