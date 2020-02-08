"""Configuration file for data file paths, extensions, load model checkpoint,
and load cached feature vectors."""
PATH_TRAIN = 'data/MiniChallenge/training'
PATH_TEST = 'data/MiniChallenge/test'
EXTENSION = '.ply'

MODEL_SELECTION = False
# whether to load a XGB file checkpoint
LOAD_TRAINED = False
# whether the features were precomputed or not for the test dataset.
TEST_FEATURES_PRECOMPUTED = False
# whether the features were precomputed or not for the validation dataset.
VAL_FEATURES_PRECOMPUTED = False
