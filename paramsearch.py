"""The entire pipeline.

Load features.
Run the XGBoost loop.
TODO: run bayesian optimization with hyperopt
"""
import numpy as np
import subprocess
from config import *
import matplotlib.pyplot as plt
import hyperopt
from utils.loader import preprocess, NAMEFILES, NAMETEST, cross_val
from sklearn.metrics import jaccard_score
from benchmark import main as run_benchmark


def validation_objective(hyperparameters):
    """Objective function :math:`f` for the Bayesian hyperparameter optimisation algorithm."""
    ## do shit
    max_depth = hyperparameters['max_depth']
    n_estimators = hyperparameters['n_estimators']
    # num_neighbors = hyperparameters['num_neighbors']
    num_neighbors = 9

    soft_labels = run_benchmark(max_depth, n_estimators)

    data_cross_val = cross_val()
    data_local = data_cross_val[0]

    # Load validation point cloud
    val_cloud, val_labels_true, _ = load_point_cloud(
        os.path.join(PATH_TRAIN, data_local['val'][0])+EXTENSION)

    g = graph.make_graph(val_cloud, n=num_neighbors)
    graph.write_graph(g, soft_labels*100, '')
    print("Created nodes and edges files.")

    run_graphcut()

    # load hard labels
    pred_hard_labels = np.loadtxt('labels.txt')
    iou_score = jaccard_score(val_labels_true, pred_hard_labels, average='macro')
    return iou_score


if __name__ == "__main__":
    
    INITIAL_HYPERPARAMS = [
        {
            'max_depth': 4,
            'n_estimators': 50
        },
        {
            'max_depth': 2,
            'n_estimators': 50
        },
        {
            'max_depth': 3,
            'n_estimators': 100
        }
    ]
    
    scores_ = []
    for hparams in INITIAL_HYPERPARAMS:
        iou_score = validation_objective(hparams)
        print("Score for params %s: %.2f" % (hparams, 100*iou_score))
        scores_.append(iou_score)
