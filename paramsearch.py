"""Parameter search using a Bayesian optimization algorithm based on
trees of Parzen estimators (TPE) i.e. hierarchial kernel density estimators.
"""
import numpy as np
from config import PATH_TRAIN, EXTENSION
import matplotlib.pyplot as plt
from utils.loader import cross_val, load_point_cloud, write_results, NAMEFILES
from utils import graph
from sklearn.metrics import jaccard_score
# TODO use right imports
from clean_benchmark import run_graphcut, predict_labels, train_paramsearch, PARAM_MODEL, VAL_PART
import os, json
import hyperopt
from hyperopt import hp
from datetime import date, datetime

LOAD_BEST = True

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


def validation_objective(hyperparameters):
    """Objective function :math:`f` for the Bayesian hyperparameter
    optimisation algorithm."""
    print('Current parameters: %s' % hyperparameters)
    max_depth = int(hyperparameters['max_depth'])
    n_estimators = int(hyperparameters['n_estimators'])
    num_neighbors = int(hyperparameters['num_neighbors'])
    alpha = hyperparameters['l1_reg']

    # TODO modify benchmark function to accept max_depth, n_estimastors, XGBoost alpha
    classifier = train_paramsearch(max_depth,n_estimators,alpha)
    preds, labels = predict_labels(PARAM_MODEL,NAMEFILES,VAL_PART,true_labels=True)

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
    iou_score = np.min(IoUs)
    print(IoUs)

    print("Obtained IoU score %.3f" % (100 * iou_score))
    return {'loss': -iou_score, 'params': hyperparameters,
            'status': hyperopt.STATUS_OK}


if __name__ == "__main__":
    plt.style.use("ggplot")
    
    space = {
        'max_depth': hp.quniform('xgb_max_depth', 2, 6, 1),
        'n_estimators': hp.quniform('xgb_n_estimators', 50, 500, 1),
        'num_neighbors': 9,
        'l1_reg': hp.lognormal('xgb_alpha', -1, 1)  # use exp(N(-1, 1)) prior
    }
    
    if LOAD_BEST:
        # with open("hyperparam_best.json", "r") as f:
        #     best_param_dic = json.load(f)
        # validation_objective(best_param_dic['result']['params'])   
        validation_objective({'l1_reg': 0.1859824127986145, 'max_depth': 5.0, 'n_estimators': 487.0, 'num_neighbors': 9})
    else:   
        maxiters = 20
        
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
        
        # save best parameters dict
        with open("hyperparam_best.json", "w") as f:
            json.dump(trials.best_trial, f, default=json_serial, indent=4)
