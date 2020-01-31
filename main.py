from loader import load_point_cloud, cross_val
from features_computation import compute_covariance_features
from subsampler import get_even_number
import os
import numpy as np 
import xgboost as xgb
import pickle

PATH = 'data/MiniChallenge/training'
EXTENSION = '.ply'
SIZE = 10000

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

        verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature = compute_covariance_features(new_train_cloud1,train_cloud1,tree1,radius=0.5)
        features_1 = np.vstack((verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature)).T
        verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature = compute_covariance_features(new_train_cloud2,train_cloud2,tree2,radius=0.5)
        features_2 = np.vstack((verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature)).T
        
        features = np.append(features_1,features_2,axis=0)
        labels = np.append(new_train_label1,new_train_label2,axis=0)

        classifier = xgb.XGBClassifier()
        classifier.fit(features,labels)
        print('Training accuracy : ' +str(classifier.score(features,labels)))

        val_cloud, val_label, tree = load_point_cloud(os.path.join(PATH,data_local['val'][0])+EXTENSION)
        verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature = compute_covariance_features(val_cloud,val_cloud,tree,radius=0.5)
        features_test = np.vstack((verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature)).T
        val_score = classifier.score(features_test,val_label)
        print('Validation accuracy : ' +str(val_score))

        classifiers.append(classifier)
        if val_score > best_score:
            best_classifier = k
            best_score = val_score
    
    pickle.dump(classifiers[best_classifier], open(r'classifier.pickle','wb'))

