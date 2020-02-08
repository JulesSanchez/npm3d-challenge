from utils.loader import load_point_cloud, cross_val, write_results, CLASSES
from utils.features_computation import compute_covariance_features, shape_distributions
from utils.subsampler import get_even_number
from sklearn.metrics import accuracy_score
import os, math
import numpy as np 
import xgboost as xgb
import tqdm
import pickle
import time
from config import *

## Feature hyperparameters
SIZE = 1000
RADIUS_COV = 0.5
MULTISCALE = [0.2,0.5,1,1.5]
RADIUS_SHAPE = 1.5
NUM_BINS = 10
PULLS = 255



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
        print('Covariance features computed. Elapsed time:', time.time()-t1)
        # print('feat cov shape:', features_cov.shape)
    
    t1 = time.time()
    A1, A2, A3, A4, D3, bins = shape_distributions(subcloud, point_cloud, tree, RADIUS_SHAPE, PULLS, NUM_BINS)
    features_shape = np.vstack((A1, A2, A3, A4, D3)).T
    if verbose:
        print('Shape features computed. Elapsed time:', time.time() - t1)
    features = np.append(features_cov, features_shape, axis=1)    
    return features


def main():
    """Main loop: assemble training data subsamples, compute features,
    run XGBoost, validate on every decimation of validation and test."""
    data_cross_val = cross_val()
    if MODEL_SELECTION:
        # Use cross validation for model selection
        classifiers = []
        best_score = 0
        best_classifier = 0

        for k in range(len(data_cross_val)):
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
                print("Subsampling time for train cloud #%d:"%(i+1), time.time() - t1)
                print("Computing features on train cloud #%d..."%(i+1))
                features = assemble_features(train_cloud, subcloud, tree)
                feature_list_.append(features)
                label_list_.append(sublabels)

            features = np.vstack(feature_list_)
            labels = np.hstack(label_list_)

            classifier = xgb.XGBClassifier(objective='multi:softprob')
            classifier.fit(features, labels)
            score_ = classifier.score(features, labels)
            print('Training accuracy: %.2f%' % (100*score_))

            val_cloud, val_label, val_tree = load_point_cloud(os.path.join(PATH_TRAIN,data_local['val'][0])+EXTENSION)
            indices = val_label > 0
            new_val_cloud = val_cloud[indices]
            new_val_label = val_label[indices]
            #Ram friendly evaluation
            soft_labels = []
            n_split = len(new_val_cloud)//100
            t1 = time.time()
            for i in range(n_split+1):
                sub_val_cloud = new_val_cloud[i*100:min((i+1)*100,len(new_val_cloud))]
                sub_features = assemble_features(new_val_cloud, sub_val_cloud, val_tree)
                soft_labels = soft_labels + list(classifier.predict_proba(sub_features))
            soft_labels = np.array(soft_labels)
            print(soft_labels.shape)
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
            write_results(os.path.join(PATH_TRAIN,data_local['val'][0]),soft_labels*100)
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
                print("Subsampling time for train cloud #%d:" %
                      (i+1), time.time() - t1)
                print("Computing features on train cloud #%d..." % (i+1))
                features = assemble_features(train_cloud, subcloud, tree)

                feature_list_.append(features)
                label_list_.append(sublabels)

            features = np.vstack(feature_list_)
            # import ipdb; ipdb.set_trace()
            labels = np.hstack(label_list_)

            classifier = xgb.XGBClassifier()
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
        n_split = len(test_cloud)//100
        print("Number of test set splits: %d" % n_split)
        t1 = time.time()
        if not TEST_FEATURES_PRECOMPUTED:
            for i in tqdm.tqdm(range(n_split+1)):
                sub_test_cloud = test_cloud[i*100:min((i+1)*100,len(test_cloud))]
                features_test = assemble_features(test_cloud, sub_test_cloud, tree, verbose=False)
                os.makedirs('features', exist_ok=True)
                np.save('features/test_'+str(i)+'.npy',features_test)
                #labels_predicted += list(classifier.predict(features_test))
                soft_labels = soft_labels + list(classifier.predict_proba(features_test))
        else :
            for i in tqdm.tqdm(range(n_split+1)):
                feature_test = np.load('features/test_'+str(i)+'.npy')
                soft_labels = soft_labels + list(classifier.predict_proba(features_test))
        soft_labels = np.array(soft_labels)
        #labels_predicted = np.array(labels_predicted)
        write_results('results/',soft_labels*100)
        with open('fullKclassifier.pickle', 'wb') as f:
            pickle.dump(classifier, f)


if __name__ == '__main__':
    main()
