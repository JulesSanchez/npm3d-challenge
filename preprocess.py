"""
Preprocess the point clouds by voxel downsampling.
"""
import os
from utils.loader import preprocess, NAMEFILES, NAMETEST

for filename in NAMEFILES:
    print("Preprocessing %s" % filename)
    basename = os.path.splitext(os.path.basename(filename))[0]
    basename += "_voxelized.ply"
    
    preprocess()


