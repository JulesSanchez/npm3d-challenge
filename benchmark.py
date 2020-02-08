from utils.loader import load_point_cloud, cross_val, write_results, CLASSES
from utils.features_computation import compute_covariance_features, shape_distributions
from utils.subsampler import get_even_number
from utils import graph
from sklearn.metrics import accuracy_score, jaccard_score
import os, math
import numpy as np 
import xgboost as xgb
import tqdm
import pickle
import time
import subprocess
from config import *

## Feature hyperparameters
SIZE = 1000
RADIUS_COV = 0.5
MULTISCALE = [0.2,0.5,1,1.5]
RADIUS_SHAPE = 1.5
NUM_BINS = 10
PULLS = 255


def voxel_downsample_data(voxel_size=0.5):
    """DEPRECATED. Preprocess the point cloud files with voxel downsampling."""
    for filename in NAMEFILES:
        filepath = os.path.join("data/MiniChallenge/training", filename)
        print("Preprocessing %s" % filepath)
        preprocess(filepath+".ply", filepath+"_voxelized", VOXEL_SIZE)

    for filename in NAMETEST:
        filepath = os.path.join("data/MiniChallenge/test", filename)
        print("Preprocessing %s" % filepath)
        preprocess(filepath+".ply", filepath +
                   "_voxelized", VOXEL_SIZE, labels=False)


def run_graphcut():
    """Run graph cut to get hard labels.
    
    Assumptions:
    - at this point in the pipeline the KNN graph nodes (with unary potentials)
    are located in a file named `nodes.txt`
    - the edges with the pairwise smoothing potentials are in `edges.txt`
    - of course the graphcut binary was built (see README)
    """
    try:
        subprocess.call("./gco/build/Main")
    except FileNotFoundError:
        subprocess.call("./build/Main")
    

def assemble_features(point_cloud: np.ndarray, subcloud: np.ndarray, tree, verbose=True):
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
    t1 = time.time()
    features_cov = []
    for radius in MULTISCALE:
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
        print('  Covariance features computed. Elapsed time: %.3f' %
              (time.time()-t1))
        # print('feat cov shape:', features_cov.shape)

    t1 = time.time()
    A1, A2, A3, A4, D3, bins = shape_distributions(
        subcloud, point_cloud, tree, RADIUS_SHAPE, PULLS, NUM_BINS)
    features_shape = np.vstack((A1, A2, A3, A4, D3)).T
    if verbose:
        print('  Shape features computed. Elapsed time: %.3f' %
              (time.time() - t1))
    features = np.append(features_cov, features_shape, axis=1)
    return features



def main(max_depth=3, n_estimators=100):
    """Main loop: assemble training data subsamples, compute features,
    run XGBoost, run on every decimation of validation and test.
    
    Parameters
    ----------
    max_depth : int (default 3)
        Maximum depth of the xgboost trees.
    n_estimators : int (default 100)
        Number of estimators for xgboost.
    
    Returns
    -------
    soft_labels : ndarray
    """
    data_cross_val = cross_val()
    if MODEL_SELECTION:
        classifiers = []
        best_score = 0
        best_classifier = 0

        # the -2 ensures only data_cross_val[0] is used
        # mind. blown.
        for k in range(len(data_cross_val)-2):
            # assemble training point cloud data
            data_local = data_cross_val[k]
            feature_list_ = []
            label_list_ = []

            for i, datafile in enumerate(data_local['training']):
                train_cloud, train_label, tree = load_point_cloud(
                    os.path.join(PATH_TRAIN, datafile+EXTENSION))
                # subsample the point cloud and labels
                t1 = time.time()
                subcloud, sublabels = get_even_number(
                    train_cloud, train_label, SIZE)
                print("Subsampling time for train cloud #%d: %.3f"%((i+1), time.time() - t1))
                print("Computing features on train cloud #%d..."%(i+1))
                features = assemble_features(train_cloud, subcloud, tree)
                features = np.hstack((features,subcloud[:,-1].reshape(-1,1)))
                feature_list_.append(features)
                label_list_.append(sublabels)

            features = np.vstack(feature_list_)
            labels = np.hstack(label_list_)

            classifier = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, objective='multi:softprob')
            classifier.fit(features, labels)
            score_ = classifier.score(features, labels)
            print('Training accuracy: %.2f' % (100*score_))

            val_cloud, val_label, val_tree = load_point_cloud(os.path.join(PATH_TRAIN,data_local['val'][0])+EXTENSION)
            indices = val_label > 0
            new_val_cloud = val_cloud[indices]
            new_val_label = val_label[indices]
            
            # Ram friendly evaluation: split the point cloud in slices
            # then compute features on each slice.
            soft_labels = []
            n_split = len(new_val_cloud)//100000
            print("Number of val set splits: %d" % n_split)
            t1 = time.time()
            if not VAL_FEATURES_PRECOMPUTED:
                print("Computing val set features.")
                for i in tqdm.tqdm(range(n_split+1)):
                    sub_val_cloud = new_val_cloud[i*100000:min((i+1)*100000,len(new_val_cloud))]
                    sub_features = assemble_features(val_cloud, sub_val_cloud, val_tree)
                    os.makedirs('features/val', exist_ok=True)
                    np.save('features/val/'+str(i)+'.npy',sub_features)
                    sub_features = np.hstack((sub_features,sub_val_cloud[:,-1].reshape(-1,1)))
                    soft_labels = soft_labels + list(classifier.predict_proba(sub_features))
            else:
                print("Loading precomputed val set features.")
                for i in tqdm.tqdm(range(n_split+1)):
                    sub_features = np.load('features/val/'+str(i)+'.npy')
                    sub_val_cloud = new_val_cloud[i*100000:min((i+1)*100000,len(new_val_cloud))]
                    sub_features = np.hstack((sub_features,sub_val_cloud[:,-1].reshape(-1,1)))
                    soft_labels = soft_labels + list(classifier.predict_proba(sub_features))
            soft_labels = np.array(soft_labels)
            labels_predicted = np.argmax(soft_labels,axis=1) + 1
            val_score = accuracy_score(new_val_label,labels_predicted)
            print('Time to score on ' +data_local['val'][0] + ' : ' + str(time.time() - t1) )
            print('Validation accuracy : ' +str(val_score))
            for i in range(1,len(CLASSES)):
                indices = new_val_label == i
                local_val_score = accuracy_score(labels_predicted[indices],new_val_label[indices])
                if math.isnan(local_val_score):
                    continue
                print('Validation accuracy for label ' + CLASSES[i] +' : '  +str(local_val_score))
            #write_results('',soft_labels*100,False)
            classifiers.append(classifier)
            if val_score > best_score:
                best_classifier = k
                best_score = val_score
        pickle.dump(classifiers[best_classifier], open(str(SIZE//1000) + 'Kclassifier.pickle','wb'))
    else:
        data_local = data_cross_val[0]
        if not LOAD_TRAINED:
            # assemble training point cloud data
            feature_list_ = []
            label_list_ = []

            for i, datafile in enumerate(data_local['training']+data_local['val']):
                train_cloud, train_label, tree = load_point_cloud(
                    os.path.join(PATH_TRAIN, datafile+EXTENSION))
                # subsample the point cloud and labels
                t1 = time.time()
                subcloud, sublabels = get_even_number(
                    train_cloud, train_label, SIZE)
                print("Subsampling time for train cloud #%d: %.3f" %
                      ((i+1), time.time() - t1))
                print("Computing features on train cloud #%d..." % (i+1))
                features = assemble_features(train_cloud, subcloud, tree)

                feature_list_.append(features)
                label_list_.append(sublabels)

            features = np.vstack(feature_list_)
            labels = np.hstack(label_list_)

            classifier = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, objective='multi:softprob')
            classifier.fit(features,labels)
            print('Training accuracy : ' +str(classifier.score(features,labels)))
            with open('fullKclassifier.pickle', 'wb') as f:
                pickle.dump(classifier, f)
        else:
            with open('fullKclassifier.pickle','rb') as f:
                classifier = pickle.load(f)

        test_cloud, tree = load_point_cloud(os.path.join(PATH_TEST,data_local['test'][0])+EXTENSION)
        #Ram friendly evaluation
        soft_labels = []
        n_split = len(test_cloud)//100000
        print("Number of test set splits: %d" % n_split)
        t1 = time.time()
        if not TEST_FEATURES_PRECOMPUTED:
            print("Computing test set features.")
            for i in tqdm.tqdm(range(n_split+1)):
                sub_test_cloud = test_cloud[i*100000:min((i+1)*100000,len(test_cloud))]
                features_test = assemble_features(test_cloud, sub_test_cloud, tree, verbose=False)
                os.makedirs('features/test', exist_ok=True)
                np.save('features/test/'+str(i)+'.npy',features_test)
                features_test = np.hstack((features_test,sub_test_cloud[:,-1].reshape(-1,1)))
                soft_labels = soft_labels + list(classifier.predict_proba(features_test))
        else:
            print("Loading precomputed test set features.")
            for i in tqdm.tqdm(range(n_split+1)):
                features_test = np.load('features/test/'+str(i)+'.npy')
                features_test = np.hstack((features_test,sub_test_cloud[:,-1].reshape(-1,1)))
                soft_labels = soft_labels + list(classifier.predict_proba(features_test))
        soft_labels = np.array(soft_labels)
        #write_results('',soft_labels*100, False)
    return soft_labels


if __name__ == '__main__':
    data_cross_val = cross_val()
    data_local = data_cross_val[0]
    if not VAL_RESULTS:
        # do not assume nodes.txt, edges.txt, and labels.txt exist
        print('Computing soft labels...')
        soft_labels = main()  # using default parameters here

        try:
            f = open("edges.txt")
            print("Edges file already exists, writing nodes file...",end=' ')
            write_results('',soft_labels*100, False)  # this only writes nodes file
            f.close()
            print("Done.")
        except :
            print("Computing graph...")
            if not MODEL_SELECTION:
                # Outside model selection, compute test set graph
                test_cloud, _ = load_point_cloud(os.path.join(PATH_TEST,data_local['test'][0])+EXTENSION)
                g = graph.make_graph(test_cloud)
            else:
                # For model selection, get validation set graph
                val_cloud, val_label, _ = load_point_cloud(os.path.join(PATH_TRAIN,data_local['val'][0])+EXTENSION)
                val_cloud = val_cloud[val_label>0]
                g = graph.make_graph(val_cloud)
            graph.write_graph(g,soft_labels*100,'')
            print("Created nodes and edges files.")
        # Get hard labels by graph cut.
        run_graphcut()
    if MODEL_SELECTION:
        _, val_label, _ = load_point_cloud(os.path.join(PATH_TRAIN,data_local['val'][0])+EXTENSION)
        val_label = val_label[val_label>0]

        
        # load soft labels for comparison
        try:
            predicted_soft_label = np.argmax(soft_labels,axis=1) + 1
        except:
            # load soft label prediction from nodes file
            predicted_soft_label = np.argmax(np.loadtxt('nodes.txt'),axis=1) + 1
        score_soft_ = jaccard_score(val_label, predicted_soft_label, average='macro')
        print("IoU before graph cut: %.2f" % (100*score_soft_))
        
        # load hard labels
        predicted_hard_label = np.loadtxt('labels.txt')
        score_hard_ = jaccard_score(val_label, predicted_hard_label, average='macro')
        print("IoU after graph cut: %.2f" % (100*score_hard_))