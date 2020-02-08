# npm3d-challenge
MiniChallenge in 2020 [NPM3D course](http://npm3d.fr/)

The data layout is:
```
data/MiniChallenge
- training
- test
```

The `0` label corresponds to `Unclassified` data points in the point clouds.


## PointNet

We use this implementation of PointNet++ in PyTorch: https://github.com/erikwijmans/Pointnet2_PyTorch

## GCO


We use "".
Grab the code:
```bash
wget http://mouse.cs.uwaterloo.ca/code/gco-v3.0.zip
```

We use CMake as a build system for the C++ code. Build the code:
```bash
mkdir build/
cd build/
cmake ../gco
make
```
You can then check that the example file compiles:
```bash
./build/Main
```


## References

* Blomley et al. _SHAPE DISTRIBUTION FEATURES FOR POINT CLOUD ANALYSIS- A GEOMETRIC HISTOGRAM APPROACH ON MULTIPLE SCALES_ https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/II-3/9/2014/isprsannals-II-3-9-2014.pdf
* Loic Landrieu, Hugo Raguet, Bruno Vallet, Clément Mallet, Martin Weinmann.  _A structured regularization framework for spatially smoothing semantic labelings of 3D point clouds._ ISPRS Journal of Photogrammetry and Remote Sensing, Elsevier, 2017, 132, pp.102-118. [Link](https://hal.archives-ouvertes.fr/hal-01505245v2)