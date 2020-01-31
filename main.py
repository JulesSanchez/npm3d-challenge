from loader import load_point_cloud, cross_val
from features_computation import compute_covariance_features
from subsampler import get_even_number
import os
import numpy as np 
import xgboost as xgb
import pickle

PATH = 'data/MiniChallenge/training'
EXTENSION = '.ply'
SIZE = 1000
RADIUS = 0.5
CLASSES = ['Unclassified','Ground','Building','Poles','Pedestrians','Cars','Vegetation']

if __name__ == '__main__':

    data_cross_val = cross_val()
    classifiers = []
    best_score = 0
    best_classifier = None

    for k in range(len(data_cross_val)):

        data_local = data_cross_val[k]
        train_cloud1, train_label1, tree1 = load_point_cloud(os.path.join(PATH,data_local['training'][0])+EXTENSION)
        train_cloud2, train_label2, tree2 = load_point_cloud(os.path.join(PATH,data_local['training'][1])+EXTENSION)
        new_train_cloud1, new_train_label1 = get_even_number(train_cloud1,train_label1,SIZE)
        new_train_cloud2, new_train_label2 = get_even_number(train_cloud2,train_label2,SIZE)

        verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature = compute_covariance_features(new_train_cloud1,train_cloud1,tree1,radius=RADIUS)
        features_1 = np.vstack((verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature)).T
        verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature = compute_covariance_features(new_train_cloud2,train_cloud2,tree2,radius=RADIUS)
        features_2 = np.vstack((verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature)).T
        
        features = np.append(features_1,features_2,axis=0)
        labels = np.append(new_train_label1,new_train_label2,axis=0)

        classifier = xgb.XGBClassifier()
        classifier.fit(features,labels)
        print('Training accuracy : ' +str(classifier.score(features,labels)))

        val_cloud, val_label, tree = load_point_cloud(os.path.join(PATH,data_local['val'][0])+EXTENSION)
        indices = val_label > 0
        new_val_cloud = val_cloud[indices]
        new_val_label = val_label[indices]
        verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature = compute_covariance_features(new_val_cloud,val_cloud,tree,radius=RADIUS)
        features_test = np.vstack((verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature)).T
        val_score = classifier.score(features_test,new_val_label)
        print('Validation accuracy : ' +str(val_score))
        for k in range(1,len(CLASSES)):
            indices = new_val_label == k
            local_val_score = classifier.score(features_test[indices],new_val_label[indices])
            print('Validation accuracy for label ' + CLASSES[k] +' : '  +str(local_val_score))


        classifiers.append(classifier)
        if val_score > best_score:
            best_classifier = k
            best_score = val_score
    
    pickle.dump(classifiers[best_classifier], open(str(SIZE) + 'Kclassifier.pickle','wb'))

