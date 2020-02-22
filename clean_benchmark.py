import os
import numpy as np
import xgboost as xgb
import pickle
import subprocess
from utils.loader import NAMEFILES, NAMETEST, write_results
from utils.features_computation import precompute_features, load_point_cloud
from utils.subsampler import get_even_number
from utils.active_learning import active_learning, train_simple, get_features, get_labels
from utils import graph
from sklearn.metrics import accuracy_score, jaccard_score
from config import EXTENSION, PATH_TRAIN, PATH_TEST
from config import LOAD_TRAINED, MODEL_SELECTION
from sklearn.metrics import accuracy_score, jaccard_score

# Feature hyperparameters
INITIAL_SIZE = 500
RADII_COV = [0.2, 0.5, 1, 1.5, 2.5]
RADII_SHAPE = [0.5, 1.5, 2.5]
N_ACTIVE = 20
N_AJOUT = 100
NAME_MODEL = "active_classifier.pickle"
BASE_MODEL = "base_classifier.pickle"
VAL_MODEL = NAME_MODEL
N_SLICE = 6
VAL_PART = [0,1]

CACHE = {}

def run_graphcut(path):
    """Run graph cut to get hard labels.
    
    Assumptions:
    - at this point in the pipeline the KNN graph nodes (with unary potentials)
    are located in a file named `nodes.txt`
    - the edges with the pairwise smoothing potentials are in `edges.txt`
    - of course the graphcut binary was built (see README)
    """
    try:
        subprocess.call(["./gco/build/Main" ,path])
    except FileNotFoundError:
        subprocess.call(["./build/Main" ,path])

def create_train_dictionnary(n_split):
    train_info = {}

    for name in NAMEFILES:
        
        train_info[name] = {}
        
        if not os.path.isdir(os.path.join('features',name)):
            precompute_features(os.path.join(PATH_TRAIN, name + EXTENSION), os.path.join('features',name), RADII_COV, RADII_SHAPE, n_slice=n_split)
        
        elts = list(range(n_split+1))
        train_info[name]['val'] = VAL_PART #np.random.choice(np.arange(n_split),1)
        train_info[name]['path'] = os.path.join('features',name)
        train_info[name]['train'] = l3 = [x for x in elts if x not in train_info[name]['val']]
        
        ind, label = get_even_number([],get_labels(train_info[name]['path'],train_info[name]['train']),size=INITIAL_SIZE,return_indices=True)
        train_info[name]['indices_train'] = ind
        train_info[name]['label_train'] = label

    return train_info

def train(max_depth=3, n_estimators=100, n_split=N_SLICE, cache=CACHE):

    classifier = xgb.XGBClassifier(max_depth=max_depth,
                                    n_estimators=n_estimators,
                                    objective='multi:softprob')

    #Make sure all features can be accessed, store relavant informations
    train_info = create_train_dictionnary(n_split)
    
    #Save simple classifier without active learning
    base_classifier = train_simple(train_info,classifier)
    with open(BASE_MODEL, 'wb') as f:
        pickle.dump(base_classifier, f)

    #Apply active learning and save resulting classifier
    active_classifier, new_dic = active_learning(train_info,base_classifier,N_ACTIVE,N_AJOUT)
    for name in new_dic:
        np.savetxt(os.path.join('features/' + name, 'indices_train.txt'),new_dic[name]['indices_train'].astype(int), fmt='%i')
    with open(NAME_MODEL, 'wb') as f:
        pickle.dump(active_classifier, f)

def predict_labels(classifier_path, names, features, n_split=N_SLICE, path=PATH_TRAIN, true_labels = None):
    
    #Load Classifier
    with open(classifier_path, 'rb') as f:
        classifier = pickle.load(f)

    predictions = []
    labels = []
    training = not true_labels is None
    #For each dataset, and relevant features name, load appropriately
    for name in names:

        if not os.path.isdir(os.path.join('features',name)):
            precompute_features(os.path.join(path, name + EXTENSION), os.path.join('features',name), RADII_COV, RADII_SHAPE, n_slice=n_split, is_train=training)

        feats = get_features(os.path.join('features',name),features)
        predictions.append(classifier.predict_proba(feats))
        if not true_labels is None:
            labels.append(get_labels(os.path.join('features',name),features))

    n_classes = predictions[0].shape[-1]
    if training:
        return predictions, labels
    else:
        return np.array(predictions).reshape(-1,n_classes)
    
if __name__ == '__main__':

    #Make classifier
    if not LOAD_TRAINED:
        train()

    #Compute val IoU
    if MODEL_SELECTION:
        preds, labels = predict_labels(VAL_MODEL,NAMEFILES,VAL_PART,true_labels=True)
        IoUs = []
        for name in NAMEFILES:
            i = NAMEFILES.index(name)
            EDGE_FILE_EXISTS = os.path.exists(name + '/'+'edges.txt')
            if EDGE_FILE_EXISTS:
                #print("Edges file already exists, writing nodes file...", end=' ')
                # this only writes nodes file
                write_results(name + '/', preds[i] * 100, False)
            else:
                os.makedirs(name, exist_ok=True)
                val_cloud, val_label, _ = load_point_cloud(
                            os.path.join(PATH_TRAIN, name) + EXTENSION)
                val_cloud = val_cloud[val_label > 0]
                len_slice = len(val_cloud)//(N_SLICE*len(VAL_PART))
                val_cloud = val_cloud[:len_slice]
                g = graph.make_graph(val_cloud)
                graph.write_graph(g, preds[i] * 100, name + '/')
            run_graphcut(name)
            predicted_hard_label = np.loadtxt(name + '/'+'labels.txt')
            IoUs.append(jaccard_score(labels[i], predicted_hard_label,
                                    average='macro'))
        print(IoUs)
        

    #Run pipeline on test set
    if True:
       preds = predict_labels(VAL_MODEL,NAMETEST,list(range(N_SLICE+1)),path=PATH_TEST,true_labels=None)
        test_cloud, _ = load_point_cloud(
            os.path.join(PATH_TEST, NAMETEST[0]) + EXTENSION)
        # g = graph.make_graph(test_cloud)
        # graph.write_graph(g, preds[0] * 100, '')
        write_results(NAMETEST[0] + '/', preds[0] * 100, False)
        run_graphcut(NAMETEST[0])


    
