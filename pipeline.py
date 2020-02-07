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


def run_graphcut():
    """Assumptions:
      - at this point in the pipeline the KNN graph nodes (with unary potentials)
      are located in a file named `nodes.txt`
      - the edges with the pairwise smoothing potentials are in `edges.txt`
      - of course the graphcut binary was built (see README)
    """
    subprocess.call("./gco/build/Main")


def objective(hyperparams):
    """Objective function :math:`f` for the Bayesian hyperparameter optimisation algorithm."""
    pass

