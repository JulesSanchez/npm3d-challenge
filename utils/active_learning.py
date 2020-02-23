import numpy as np 
import os
EPSILON = 10e-10

def get_labels(path,list_labels, indices = None):
    labels = []
    for i in list_labels:
        labels.append(np.load(os.path.join(path,str(i) + '_labels.npy')))
    if indices is None:
        return np.hstack(labels)
    else :
        return np.hstack(labels)[indices]

def get_features(path, list_features, indices = None):
    features = []
    for i in list_features:
        features.append(np.load(os.path.join(path, str(i) + '.npy')))
    if indices is None:
        return np.vstack(features)
    else :
        return np.vstack(features)[indices]

def train_simple(dic_info, classifier):
    features = []
    labels = []
    for name in dic_info:
        features.append(get_features(dic_info[name]['path'],dic_info[name]['train'],dic_info[name]['indices_train']))
        labels.append(dic_info[name]['label_train'])
    features = np.vstack(features)
    labels = np.hstack(labels)
    classifier.fit(features, labels)
    return classifier

def get_new_indices(dic_info, classifier, n_indices=100):
    for name in dic_info:
        feat = get_features(dic_info[name]['path'],dic_info[name]['train'])
        proba = classifier.predict_proba(feat)
        best_pred = np.sum(proba*np.log(proba+EPSILON),axis=1)
        i = 0
        j = 0
        indices = best_pred.argsort()
        ind_to_append = []
        while i < n_indices and i < len(indices):
            if not indices[j] in dic_info[name]['indices_train']:
                ind_to_append.append(indices[j])
                i += 1
            j += 1
        dic_info[name]['indices_train'] = np.append(dic_info[name]['indices_train'],np.array(ind_to_append))
        dic_info[name]['label_train'] = get_labels(dic_info[name]['path'],dic_info[name]['train'])[dic_info[name]['indices_train']]
    return dic_info

def active_learning(dic_info, classifier, n_repeats = 50, n_indices=20):
    for k in range(n_repeats):
        dic_info = get_new_indices(dic_info, classifier, n_indices)
        classifier = train_simple(dic_info, classifier)
        print("One pass of active learning done.")
    return classifier, dic_info