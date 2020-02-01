from loader import load_point_cloud, cross_val
from features_computation import compute_covariance_features, shape_distributions
from subsampler import get_even_number
import os, math
import numpy as np 
import xgboost as xgb
import pickle

PATH = 'data/MiniChallenge/training'
PATH_TEST = 'data/MiniChallenge/test'
EXTENSION = '.ply'
SIZE = 1000
RADIUS_COV = 0.5
RADIUS_SHAPE = 1.5
BINS = 10
PULLS = 255
CLASSES = ['Unclassified','Ground','Building','Poles','Pedestrians','Cars','Vegetation']
import time 

if __name__ == '__main__':

    data_cross_val = cross_val()
    classifiers = []
    best_score = 0
    best_classifier = 0

    for k in range(len(data_cross_val)):

        data_local = data_cross_val[k]
        train_cloud1, train_label1, tree1 = load_point_cloud(os.path.join(PATH,data_local['training'][0])+EXTENSION)
        train_cloud2, train_label2, tree2 = load_point_cloud(os.path.join(PATH,data_local['training'][1])+EXTENSION)
        t1 = time.time()
        new_train_cloud1, new_train_label1 = get_even_number(train_cloud1,train_label1,SIZE)
        print('Time to subsample ' +data_local['training'][0] + ' : ' + str(time.time() - t1) )
        t1 = time.time()
        new_train_cloud2, new_train_label2 = get_even_number(train_cloud2,train_label2,SIZE)
        print('Time to subsample ' +data_local['training'][1] + ' : ' + str(time.time() - t1) )

        t1 = time.time()

        verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature = compute_covariance_features(new_train_cloud1,train_cloud1,tree1,radius=RADIUS_COV)
        print('Time to compute Cov Features for ' +data_local['training'][0] + ' : ' + str(time.time() - t1) )

        print('Cov Features 1 computed')
        t1 = time.time()
        A1, A2, A3, A4, D3, BINS = shape_distributions(new_train_cloud1,train_cloud1,tree1,RADIUS_SHAPE,PULLS,BINS)
        print('Time to compute Shape Features for ' +data_local['training'][0] + ' : ' + str(time.time() - t1) )
        print('Shape Features 1 computed')
        features_1_cov = np.vstack((verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature)).T
        features_1_shape = np.vstack((A1, A2, A3, A4, D3)).T
        features_1 = np.append(features_1_cov, features_1_shape,axis=1)
        
        t1 = time.time()
        verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature = compute_covariance_features(new_train_cloud2,train_cloud2,tree2,radius=RADIUS_COV)
        print('Time to compute Cov Features for ' +data_local['training'][1] + ' : ' + str(time.time() - t1) )
        print('Cov Features 2 computed')
        t1 = time.time()
        A1, A2, A3, A4, D3, _ = shape_distributions(new_train_cloud2,train_cloud2,tree2,RADIUS_SHAPE,PULLS,BINS)
        print('Time to compute Shape Features for ' +data_local['training'][1] + ' : ' + str(time.time() - t1) )
        print('Shape Features 2 computed')
        features_2_cov = np.vstack((verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature)).T
        features_2_shape = np.vstack((A1, A2, A3, A4, D3)).T
        features_2 = np.append(features_2_cov, features_2_shape,axis=1)
        
        features = np.append(features_1,features_2,axis=0)
        print(features.shape)
        labels = np.append(new_train_label1,new_train_label2,axis=0)

        classifier = xgb.XGBClassifier()
        classifier.fit(features,labels)
        print('Training accuracy : ' +str(classifier.score(features,labels)))

        val_cloud, val_label, tree = load_point_cloud(os.path.join(PATH,data_local['val'][0])+EXTENSION)
        indices = val_label > 0
        new_val_cloud = val_cloud[indices]
        new_val_label = val_label[indices]
        verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature = compute_covariance_features(new_val_cloud,val_cloud,tree,radius=RADIUS_COV)
        A1, A2, A3, A4, D3, _ = shape_distributions(new_val_cloud,val_cloud,tree,RADIUS_SHAPE,PULLS,BINS)
        features_test_cov = np.vstack((verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature)).T
        features_test_shape = np.vstack((A1, A2, A3, A4, D3)).T
        features_test = np.append(features_test_cov, features_test_shape,axis=1)
        val_score = classifier.score(features_test,new_val_label)
        print('Validation accuracy : ' +str(val_score))
        for i in range(1,len(CLASSES)):
            indices = new_val_label == i
            local_val_score = classifier.score(features_test[indices],new_val_label[indices])
            if math.isnan(local_val_score):
                continue
            print('Validation accuracy for label ' + CLASSES[i] +' : '  +str(local_val_score))


        classifiers.append(classifier)
        if val_score > best_score:
            best_classifier = k
            best_score = val_score
        break
    
    pickle.dump(classifiers[best_classifier], open(str(SIZE/1000) + 'Kclassifier.pickle','wb'))

