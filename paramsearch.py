"""Parameter search using a Bayesian optimization algorithm based on
trees of Parzen estimators (TPE) i.e. hierarchial kernel density estimators.
"""
import numpy as np
from config import PATH_TRAIN, EXTENSION
import matplotlib.pyplot as plt
from utils.loader import cross_val, load_point_cloud, write_results
from utils import graph
from sklearn.metrics import jaccard_score
from benchmark import run_graphcut, main as run_benchmark, CACHE
import os
import hyperopt
from hyperopt import hp


def validation_objective(hyperparameters):
    """Objective function :math:`f` for the Bayesian hyperparameter
    optimisation algorithm."""
    print('Current parameters: %s' % hyperparameters)
    max_depth = hyperparameters['max_depth']
    n_estimators = hyperparameters['n_estimators']
    num_neighbors = hyperparameters['num_neighbors']

    soft_labels = run_benchmark(max_depth, n_estimators)

    data_cross_val = cross_val()
    data_local = data_cross_val[0]

    # Load validation point cloud
    val_cloud, val_labels, _ = load_point_cloud(
        os.path.join(PATH_TRAIN, data_local['val'][0]) + EXTENSION)
    val_cloud = val_cloud[val_labels > 0]
    val_labels = val_labels[val_labels > 0]

    EDGE_FILE_EXISTS = os.path.exists("edges.txt")
    if EDGE_FILE_EXISTS:
        # this only writes nodes file
        write_results('', soft_labels * 100, False)
    else:
        g = graph.make_graph(val_cloud, n=num_neighbors)
        graph.write_graph(g, soft_labels * 100, '')
    print("Created nodes and edges files.")

    run_graphcut()

    # load hard labels
    pred_hard_labels = np.loadtxt('labels.txt')
    iou_score = jaccard_score(val_labels, pred_hard_labels,
                              average='macro')
    print("Obtained IoU score %.3f" % (100 * iou_score))
    return -iou_score


if __name__ == "__main__":
    plt.style.use("ggplot")
    
    space = {
        'max_depth': 2 + hp.randint('xgb_max_depth', 4),
        'n_estimators': 50 + hp.randint('xgb_n_estimators', 450),
        'num_neighbors': 9
    }
    
    maxiters = 10
    
    trials = hyperopt.Trials()
    
    best = hyperopt.fmin(validation_objective,
                         space=space,
                         algo=hyperopt.tpe.suggest,
                         max_evals=maxiters,
                         trials=trials,
                         show_progressbar=False)
    
    plt.plot(-np.asarray(trials.losses()))
    plt.title("Evolution of IoU during Bayesian optimisation")
    plt.show()

    import json
    
    # save best parameters dict
    with open("hyperparam_best.json") as f:
        json.dump(best, f)

