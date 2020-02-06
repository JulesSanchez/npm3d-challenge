# npm3d-challenge
MiniChallenge in 2020 [NPM3D course](http://npm3d.fr/)

The data layout is:
```
data/MiniChallenge
- training
- test
```

The `0` label corresponds to `Unclassified` data points in the point clouds.

## Feature computations

We wrote some of the feature computation in Cython (with little speedup)
```
python setup.py build_ext -i
```

## PointNet

We use this implementation of PointNet++ in PyTorch: https://github.com/erikwijmans/Pointnet2_PyTorch

## GCO


We use "".
Grab the code:
```bash
wget http://mouse.cs.uwaterloo.ca/code/gco-v3.0.zip
```


## References

* Blomley et al. _SHAPE DISTRIBUTION FEATURES FOR POINT CLOUD ANALYSIS- A GEOMETRIC HISTOGRAM APPROACH ON MULTIPLE SCALES_ https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/II-3/9/2014/isprsannals-II-3-9-2014.pdf
