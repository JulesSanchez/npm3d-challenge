import numpy as np 
from loader import get_features, get_labels

EPSILON = 10e-10

def train_simple(dic_info, classifier):
    features = []
    labels = []
    for name in dic_info:
        features.append(get_features(train_info[name]['path'],train_info[name]['train'],train_info[name]['indices_train']))
        labels.append(train_info[name]['label_train'])
    features = np.vstack(features)
    labels = np.hstack(labels)
    classifier.fit(features, labels)
    return classifier

def get_new_indices(dic_info, classifier, n_indices=100):
    for name in dic_info:
        feat = get_features(train_info[name]['path'],train_info[name]['train'])
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
        dic_info[name]['indices_train'] = np.append(ind,np.array(ind_to_append))
        dic_info[name]['label_train'] = get_labels(train_info[name]['path'],train_info[name]['train'])[dic_info[name]['indices_train']]
    return dic_info

def active_learning(dic_info, classifier, n_repeats = 50, n_indices=20):
    for _ in range(n_repeats):
        dic_info = get_new_indices(dic_info, classifier)
        classifier = train_simple(dic_info, classifier, n_indices)
    return classifier