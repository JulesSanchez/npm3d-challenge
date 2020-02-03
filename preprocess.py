"""
Preprocess the point clouds by voxel downsampling.
"""
import os
from utils.loader import preprocess, NAMEFILES, NAMETEST

VOXEL_SIZE = 0.5

for filename in NAMEFILES:
    filepath = os.path.join("data/MiniChallenge/training", filename)
    print("Preprocessing %s" % filepath)
    preprocess(filepath+".ply", filepath+"_voxelized", VOXEL_SIZE)

for filename in NAMETEST:
    filepath = os.path.join("data/MiniChallenge/test", filename)
    print("Preprocessing %s" % filepath)
    preprocess(filepath+".ply", filepath+"_voxelized", VOXEL_SIZE, labels=False)
