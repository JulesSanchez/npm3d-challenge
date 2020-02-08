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
from utils.loader import preprocess, NAMEFILES, NAMETEST
from sklearn.metrics import jaccard_score


def voxel_downsample_data(voxel_size=0.5):
    """DEPRECATED. Preprocess the point cloud files with voxel downsampling."""
    for filename in NAMEFILES:
        filepath = os.path.join("data/MiniChallenge/training", filename)
        print("Preprocessing %s" % filepath)
        preprocess(filepath+".ply", filepath+"_voxelized", VOXEL_SIZE)

    for filename in NAMETEST:
        filepath = os.path.join("data/MiniChallenge/test", filename)
        print("Preprocessing %s" % filepath)
        preprocess(filepath+".ply", filepath +
                   "_voxelized", VOXEL_SIZE, labels=False)


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




