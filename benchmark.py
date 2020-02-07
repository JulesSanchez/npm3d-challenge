from utils.loader import load_point_cloud, cross_val, write_results, CLASSES
from utils.features_computation import compute_covariance_features, shape_distributions
from utils.subsampler import get_even_number
from sklearn.metrics import accuracy_score
import os, math
import numpy as np 
import xgboost as xgb
import pickle
import time 

PATH_TRAIN = 'data/MiniChallenge/training'
PATH_TEST = 'data/MiniChallenge/test'
EXTENSION = '.ply'

## Feature hyperparameters
SIZE = 1000
RADIUS_COV = 0.5
MULTISCALE = [0.2,0.5,1,1.5]
RADIUS_SHAPE = 1.5
BINS = 10
PULLS = 255
MODEL_SELECTION = False

# whether to load a XGB file checkpoint
COMPUTED = True
# whether the features were precomputed or not for the test dataset.
TEST_FEATURES_PRECOMPUTED = False


def assemble_features(point_cloud: np.ndarray, subcloud: np.ndarray, tree):
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
    features_cov = np.empty((point_cloud.shape[0], 0), dtype=np.float)
    for radius in MULTISCALE:
        verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature = compute_covariance_features(
            point_cloud, subcloud, tree, radius=radius)
        # Assemble local covariance features.
        features_cov_local = np.vstack(
            (verticality, linearity, planarity,
             sphericity, omnivariance, anisotropy,
             eigenentropy, sumeigen, change_curvature)
            ).T
        # Add to multi-scale list of covariance features.
        features_cov.append(features_cov_local)
    # Stack all covariance features.
    features_cov = np.stack(features_cov, axis=1)
    print('Covariance features computed. Elapsed time:', time.time()-t1)
    
    t1 = time.time()
    A1, A2, A3, A4, D3, BINS = shape_distributions(point_cloud, subcloud, tree, RADIUS_SHAPE, PULLS, BINS)
    features_1_shape = np.vstack((A1, A2, A3, A4, D3)).T
    print('Shape features computed. Elapsed time:', time.time() - t1)
    features = np.append(features_cov, features_shape, axis=1)    
    return features


if __name__ == '__main__':

    data_cross_val = cross_val()
    if MODEL_SELECTION:
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
                print("Subsampling time for train cloud #%d:"%i, time.time() - t1)
                print("Computing features on train cloud #%d..."%i)
                features = assemble_features(train_cloud, subcloud, tree)
                
                feature_list_.append(features)
                label_list_.append(new_train_label)


            features = np.stack(feature_list_, axis=0)
            labels = np.stack(label_list_, axis=0)

            # train_cloud1, train_label1, tree1 = load_point_cloud(os.path.join(PATH_TRAIN, data_local['training'][0])+EXTENSION)
            # train_cloud2, train_label2, tree2 = load_point_cloud(os.path.join(PATH_TRAIN,data_local['training'][1])+EXTENSION)
            # # subsample the point cloud and labels
            # t1 = time.time()
            # subcloud1, sublabels1 = get_even_number(train_cloud1, train_label1, SIZE)
            # print("Subsampling time for train cloud #1:", time.time() - t1)
            # print("Computing features on train cloud #1...")
            # features_1 = assemble_features(train_cloud1, subcloud1, tree1)

            # t1 = time.time()
            # subcloud2, sublabels2 = get_even_number(train_cloud2, train_label2, SIZE)
            # print("Subsampling time for train cloud #1:", time.time() - t1)
            # print("Computing features on train cloud #2...")
            # features_2 = assemble_features(train_cloud2, subcloud2, tree2)
            
            # features = np.append(features_1, features_2, axis=0)
            # labels = np.append(new_train_label1, new_train_label2,axis=0)

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
                # features_test_cov = np.empty((len(sub_val_cloud),0),float)
                # for radi in MULTISCALE:
                #     verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature = compute_covariance_features(sub_val_cloud,val_cloud,tree,radius=radi)
                #     features_test_cov_local = np.vstack((verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature)).T
                #     features_test_cov = np.append(features_test_cov,features_test_cov_local,axis=1)
                # A1, A2, A3, A4, D3, _ = shape_distributions(sub_val_cloud,val_cloud,tree,RADIUS_SHAPE,PULLS,BINS)
                # features_test_shape = np.vstack((A1, A2, A3, A4, D3)).T
                # features_test = np.append(features_test_cov, features_test_shape,axis=1)
                # soft_labels = soft_labels + list(classifier.predict_proba(features_test))
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
    
    else :
        data_local = data_cross_val[0]
        if not COMPUTED:
            # assemble training point cloud data
            train_cloud1, train_label1, tree1 = load_point_cloud(os.path.join(PATH_TRAIN,data_local['training'][0])+EXTENSION)
            train_cloud2, train_label2, tree2 = load_point_cloud(os.path.join(PATH_TRAIN,data_local['training'][1])+EXTENSION)
            train_cloud3, train_label3, tree3 = load_point_cloud(os.path.join(PATH_TRAIN,data_local['val'][0])+EXTENSION)
            new_train_cloud1, new_train_label1 = get_even_number(train_cloud1,train_label1,SIZE)
            new_train_cloud2, new_train_label2 = get_even_number(train_cloud2,train_label2,SIZE)
            new_train_cloud3, new_train_label3 = get_even_number(train_cloud3,train_label3,SIZE)

            t1 = time.time()
            features_1_cov = np.empty((len(new_train_cloud1),0),float)
            for radi in MULTISCALE:
                verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature = compute_covariance_features(new_train_cloud1,train_cloud1,tree1,radius=radi)
                features_1_cov_local = np.vstack((verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature)).T
                features_1_cov = np.append(features_1_cov,features_1_cov_local,axis=1)
            print('Time to compute Cov Features for ' +data_local['training'][0] + ' : ' + str(time.time() - t1) )
            print('Cov Features 1 computed')
            t1 = time.time()
            A1, A2, A3, A4, D3, BINS = shape_distributions(new_train_cloud1,train_cloud1,tree1,RADIUS_SHAPE,PULLS,BINS)
            features_1_shape = np.vstack((A1, A2, A3, A4, D3)).T
            print('Time to compute Shape Features for ' +data_local['training'][0] + ' : ' + str(time.time() - t1) )
            print('Shape Features 1 computed')
            features_1 = np.append(features_1_cov, features_1_shape,axis=1)
            
            t1 = time.time()
            features_2_cov = np.empty((len(new_train_cloud2),0),float)
            for radi in MULTISCALE:
                verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature = compute_covariance_features(new_train_cloud2,train_cloud2,tree2,radius=radi)
                features_2_cov_local = np.vstack((verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature)).T
                features_2_cov = np.append(features_2_cov,features_2_cov_local,axis=1)
            print('Time to compute Cov Features for ' +data_local['training'][1] + ' : ' + str(time.time() - t1) )
            print('Cov Features 2 computed')
            t1 = time.time()
            A1, A2, A3, A4, D3, _ = shape_distributions(new_train_cloud2,train_cloud2,tree2,RADIUS_SHAPE,PULLS,BINS)
            features_2_shape = np.vstack((A1, A2, A3, A4, D3)).T
            print('Time to compute Shape Features for ' +data_local['training'][1] + ' : ' + str(time.time() - t1) )
            print('Shape Features 2 computed')
            features_2 = np.append(features_2_cov, features_2_shape,axis=1)
            
            t1 = time.time()
            features_3_cov = np.empty((len(new_train_cloud3),0),float)
            for radi in MULTISCALE:
                verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature = compute_covariance_features(new_train_cloud3,train_cloud3,tree3,radius=radi)
                features_3_cov_local = np.vstack((verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature)).T
                features_3_cov = np.append(features_3_cov,features_3_cov_local,axis=1)
            print('Time to compute Cov Features for ' +data_local['val'][0] + ' : ' + str(time.time() - t1) )
            print('Cov Features 2 computed')
            t1 = time.time()
            A1, A2, A3, A4, D3, _ = shape_distributions(new_train_cloud3,train_cloud3,tree3,RADIUS_SHAPE,PULLS,BINS)
            features_3_shape = np.vstack((A1, A2, A3, A4, D3)).T
            print('Time to compute Shape Features for ' +data_local['val'][0] + ' : ' + str(time.time() - t1) )
            print('Shape Features 2 computed')
            features_3 = np.append(features_3_cov, features_3_shape,axis=1)

            features = np.append(features_1,features_2,axis=0)
            features = np.append(features,features_3,axis=0)
            labels = np.append(new_train_label1,new_train_label2,axis=0)
            labels = np.append(labels,new_train_label3,axis=0)

            classifier = xgb.XGBClassifier()
            classifier.fit(features,labels)
            print('Training accuracy : ' +str(classifier.score(features,labels)))
            classifier = pickle.dump(classifier,open('fullKclassifier.pickle','wb'))
        else :
            classifier = pickle.load(open('fullKclassifier.pickle','rb'))

        test_cloud, tree = load_point_cloud(os.path.join(PATH_TEST,data_local['test'][0])+EXTENSION)
        #Ram friendly evaluation
        soft_labels = []
        n_split = len(test_cloud)//100
        t1 = time.time()
        if not TEST_FEATURES_PRECOMPUTED:
            for i in range(780,n_split+1):
                sub_test_cloud = test_cloud[i*100:min((i+1)*100,len(test_cloud))]
                features_test_cov = np.empty((len(sub_test_cloud),0),float)
                for radi in MULTISCALE:
                    verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature = compute_covariance_features(sub_test_cloud, test_cloud, tree,radius=radi)
                    features_test_cov_local = np.vstack((verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature)).T
                    features_test_cov = np.append(features_test_cov,features_test_cov_local,axis=1)
                A1, A2, A3, A4, D3, _ = shape_distributions(sub_test_cloud,test_cloud,tree,RADIUS_SHAPE,PULLS,BINS)
                features_test_shape = np.vstack((A1, A2, A3, A4, D3)).T
                features_test = np.append(features_test_cov, features_test_shape,axis=1)
                np.save('features/test_'+str(i)+'.npy',features_test)
                #labels_predicted += list(classifier.predict(features_test))
                soft_labels = soft_labels + list(classifier.predict_proba(features_test))
        else :
            for i in range(n_split+1):
                feature_test = np.load('features/test_'+str(i)+'.npy')
                soft_labels = soft_labels + list(classifier.predict_proba(features_test))
        soft_labels = np.array(soft_labels)
        #labels_predicted = np.array(labels_predicted)
        write_results('results/',soft_labels*100)
        pickle.dump(classifier, open('fullKclassifier.pickle','wb'))
