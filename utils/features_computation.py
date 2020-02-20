
import numpy as np
# Import functions from scikit-learn
from sklearn.neighbors import KDTree
import time
import numba
from loader import load_point_cloud
EPSILON = 1e-10 

@numba.njit
def tetrahedron_calc_volume(a, b, c, d):
    n = a.shape[0]
    mat = np.stack(((a-b).T, (b-c).T, (c-d).T)).T
    out = np.zeros((n,))
    
    for i in range(n):
        out[i] = np.abs(np.linalg.det(mat[i]))/6
    # test = 
    return out

@numba.njit
def area(a, b, c):
    n = a.shape[0]
    out = np.zeros((n,))
    for i in range(n):
        out[i] = np.linalg.norm(np.cross(b[i]-a[i], c[i]-a[i]))
    return 0.5 * out

def get_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.diag(np.inner(ba, bc)) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle

@numba.njit
def adaptative_bin(samples,bins):
    total_number = len(samples)
    bin_s = [np.percentile(samples,i*10) for i in range(11)]
    return np.array(bin_s)


@numba.jit
def local_PCA(points):
    barycenter = np.array([np.mean(points[:,i]) for i in range(3)])
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

def assemble_features(point_cloud: np.ndarray, subcloud: np.ndarray, tree, scales_cov, scales_shape, verbose=True):
    """Extract and assemble a feature vector for the point cloud.
    
    Parameters
    ----------
    point_cloud : ndarray
        Point cloud data in R^3.
    subcloud : ndarray
        Subset of the point cloud.
    tree : KDTree
        Point cloud KDTree used to compute features using nearest-neighbor searches.
    
    Returns
    -------
    features : ndarray
        Combined vector of all features.
    """
    NUM_BINS = 10
    PULLS = 255
    t1 = time.time()
    features_cov = []
    for radius in scales_cov:
        verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature = compute_covariance_features(
            subcloud, point_cloud, tree, radius=radius)
        # Assemble local covariance features.
        features_cov_local = np.vstack(
            (verticality, linearity, planarity,
             sphericity, omnivariance, anisotropy,
             eigenentropy, sumeigen, change_curvature)
        ).T
        # Add to multi-scale list of covariance features.
        features_cov.append(features_cov_local)
    # Stack all covariance features.
    features_cov = np.hstack(features_cov)
    if verbose:
        print('  Covariance features computed in time %.2f' %
              (time.time() - t1), end=' ')
        # print('feat cov shape:', features_cov.shape)

    t1 = time.time()
    features_shape = []
    for radius in scales_shape:
        A1, A2, A3, A4, D3, bins = shape_distributions(
            subcloud, point_cloud, tree, scales_shape, PULLS, NUM_BINS)
        features_shape_local = np.vstack((A1, A2, A3, A4, D3)).T
        # Add to multi-scale list of covariance features.
        features_shape.append(features_shape_local)
    features_shape = np.hstack(features_shape)
    if verbose:
        print('  Shape features computed in time: %.2f' %
              (time.time() - t1))
    features = np.append(features_cov, features_shape, axis=1)
    return features

def precompute_features(path,save_path,cov_scale,shape_scale,is_train=True,n_slice=3)
    if is_train:
        cloud, label, tree = load_point_cloud(path)
        local_cloud = cloud[label>0]
        local_label = label[label>0]
    else:
        cloud, tree = load_point_cloud(path)
        local_cloud = cloud
    # Ram friendly evaluation: split the point cloud in n_slice slices
    # then compute features on each slice.
    print("Computing features.")
    len_slice = len(local_cloud)//n_slice
    for i in tqdm.tqdm(range(n_slice+1)):
        sub_cloud = local_cloud[i * len_slice:min((i + 1) * len_slice, len(cloud))]
        sub_features = assemble_features(cloud, sub_cloud, tree, cov_scale, shape_scale)
        sub_features = np.hstack((sub_features, sub_cloud[:, -1].reshape(-1, 1)))
        os.makedirs(save_path, exist_ok=True)
        np.save(save_path + str(i) + '.npy', sub_features)
        if is_train:
            sub_labels = local_label[i * len_slice:min((i + 1) * len_slice, len(cloud))]
            np.save(save_path + str(i) + '_labels.npy', sub_labels)
    return len_slice

def tda_features(point_cloud: np.ndarray):
    """Extract more features using Topological Data Analysis (TDA).
    TODO: think about what features to use. Maybe work more locally?
    """
    import gtda.homology as hl
    homology_dims = [0, 1]
    peristence = hl.VietorisRipsPersistence(
        metric="euclidean",
        homology_dimensions=homology_dims
    )
    diagram = peristence.fit_transform(point_cloud)

    pass

