"""
The entire pipeline.

Load features.
Run the XGBoost loop.
TODO: run bayesian optimization with hyperopt
"""
import numpy as np
import matplotlib.pyplot as plt
import hyperopt
import subprocess
from sklearn.metrics import jaccard_score


def run_graphcut():
    """Assumptions:
    - at this point in the pipeline the KNN graph nodes (with unary potentials)
    are located in a file named `nodes.txt`
    - the edges with the pairwise smoothing potentials are in `edges.txt`
    - of course the graphcut binary was built (see README)
    """
    subprocess.call("./gco/build/Main")


def main(hyperparams):
    """Objective function :math:`f` for the Bayesian hyperparameter optimisation algorithm."""
    ## do shit


    labels_true = None
    
    
    # Run the Graph cut smoothing
    run_graphcut()
    # load smoothed labels
    smoothed_predlabels = np.loadtxt("labels.txt")

    iou_score = jaccard_score(labels_true, smoothed_predlabels)

    pass




