"""Configuration file for data file paths, extensions, load model checkpoint,
and load cached feature vectors."""
PATH_TRAIN = 'data/MiniChallenge/training'
PATH_TEST = 'data/MiniChallenge/test'
EXTENSION = '.ply'

# evaluate model on validation set
MODEL_SELECTION = True
# whether to load a XGB file checkpoint
LOAD_TRAINED = False
# whether the features were precomputed or not for the test dataset.
TEST_FEATURES_PRECOMPUTED = True
# whether the features were precomputed or not for the validation dataset.
VAL_FEATURES_PRECOMPUTED = True
# whether to evaluate validation results assuming nodes, labels and edges exist
VAL_RESULTS = False
