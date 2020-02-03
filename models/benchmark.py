from utils.loader import load_point_cloud, cross_val, write_results
from utils.features_computation import compute_covariance_features, shape_distributions
from utils.subsampler import get_even_number
from sklearn.metrics import accuracy_score
import os, math
import numpy as np 
import xgboost as xgb
import pickle

PATH = 'data/MiniChallenge/training'
PATH_TEST = 'data/MiniChallenge/test'
EXTENSION = '.ply'
SIZE = 1000
RADIUS_COV = 0.5
MULTISCALE = [0.2,0.5,1,1.5]
RADIUS_SHAPE = 1.5
BINS = 10
PULLS = 255
CLASSES = ['Unclassified','Ground','Building','Poles','Pedestrians','Cars','Vegetation']
MODEL_SELECTION = False 
import time 

if __name__ == '__main__':

    data_cross_val = cross_val()
    if MODEL_SELECTION:
        classifiers = []
        best_score = 0
        best_classifier = 0

        for k in range(len(data_cross_val) - 1):
            # assemble training point cloud data
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
            
            features = np.append(features_1,features_2,axis=0)
            labels = np.append(new_train_label1,new_train_label2,axis=0)

            classifier = xgb.XGBClassifier()
            classifier.fit(features,labels)
            print('Training accuracy : ' +str(classifier.score(features,labels)))

            val_cloud, val_label, tree = load_point_cloud(os.path.join(PATH,data_local['val'][0])+EXTENSION)
            indices = val_label > 0
            new_val_cloud = val_cloud[indices]
            new_val_label = val_label[indices]
            #Ram friendly evaluation
            labels_predicted = []
            n_split = len(new_val_cloud)//100
            t1 = time.time()
            for i in range(n_split+1):
                local_val_cloud = new_val_cloud[i*100:min((i+1)*100,len(new_val_cloud))]
                features_test_cov = np.empty((len(local_val_cloud),0),float)
                for radi in MULTISCALE:
                    verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature = compute_covariance_features(local_val_cloud,val_cloud,tree,radius=radi)
                    features_test_cov_local = np.vstack((verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature)).T
                    features_test_cov = np.append(features_test_cov,features_test_cov_local,axis=1)
                A1, A2, A3, A4, D3, _ = shape_distributions(local_val_cloud,val_cloud,tree,RADIUS_SHAPE,PULLS,BINS)
                features_test_shape = np.vstack((A1, A2, A3, A4, D3)).T
                features_test = np.append(features_test_cov, features_test_shape,axis=1)
                labels_predicted += list(classifier.predict(features_test))
            labels_predicted = np.array(labels_predicted)
            val_score = accuracy_score(new_val_label,labels_predicted)
            print('Time to score on ' +data_local['val'][0] + ' : ' + str(time.time() - t1) )
            print('Validation accuracy : ' +str(val_score))
            for i in range(1,len(CLASSES)):
                indices = new_val_label == i
                local_val_score = accuracy_score(labels_predicted[indices],new_val_label[indices])
                if math.isnan(local_val_score):
                    continue
                print('Validation accuracy for label ' + CLASSES[i] +' : '  +str(local_val_score))

            classifiers.append(classifier)
            if val_score > best_score:
                best_classifier = k
                best_score = val_score
        
        pickle.dump(classifiers[best_classifier], open(str(SIZE//1000) + 'Kclassifier.pickle','wb'))
    
    else :
        for k in range(len(data_cross_val) - 1):
            # assemble training point cloud data
            data_local = data_cross_val[k]
            train_cloud1, train_label1, tree1 = load_point_cloud(os.path.join(PATH,data_local['training'][0])+EXTENSION)
            train_cloud2, train_label2, tree2 = load_point_cloud(os.path.join(PATH,data_local['training'][1])+EXTENSION)
            train_cloud3, train_label3, tree3 = load_point_cloud(os.path.join(PATH,data_local['val'][0])+EXTENSION)
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


            test_cloud, tree = load_point_cloud(os.path.join(PATH_TEST,data_local['test'][0])+EXTENSION)
            #Ram friendly evaluation
            labels_predicted = []
            n_split = len(test_cloud)//100
            t1 = time.time()
            for i in range(n_split+1):
                local_val_cloud = test_cloud[i*100:min((i+1)*100,len(test_cloud))]
                features_test_cov = np.empty((len(local_val_cloud),0),float)
                for radi in MULTISCALE:
                    verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature = compute_covariance_features(local_val_cloud,test_cloud,tree,radius=radi)
                    features_test_cov_local = np.vstack((verticality, linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, sumeigen, change_curvature)).T
                    features_test_cov = np.append(features_test_cov,features_test_cov_local,axis=1)
                A1, A2, A3, A4, D3, _ = shape_distributions(local_val_cloud,test_cloud,tree,RADIUS_SHAPE,PULLS,BINS)
                features_test_shape = np.vstack((A1, A2, A3, A4, D3)).T
                features_test = np.append(features_test_cov, features_test_shape,axis=1)
                labels_predicted += list(classifier.predict(features_test))
            labels_predicted = np.array(labels_predicted)
            write_results('results/',labels_predicted)
            pickle.dump(classifier, open('fullKclassifier.pickle','wb'))