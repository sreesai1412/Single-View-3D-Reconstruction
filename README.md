# Self-Supervised Single View 3D-Reconstruction

### Requiremets
Python 3 \
PyTorch >= 1.4.0

### Dependencies

#### SoftRas renderer
Install the [SoftRas](https://arxiv.org/abs/1904.01786) renderer
```
git clone https://github.com/ShichenLiu/SoftRas.git
cd SoftRas
python setup.py install
```
#### Chamfer distance
Install [this](https://github.com/krrish94/chamferdist) PyTorch module to compute Chamfer distance between two point clouds
```
git clone https://github.com/krrish94/chamferdist
cd chamferdist
python setup.py install
```
#### Perceptual Similarity
Install the Perceptual Similarity loss
```
git clone https://github.com/shubhtuls/PerceptualSimilarity.git
```

### Other pre-requisites

#### CUB Data
Download CUB-200-2011 images
```
cd misc && wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz && tar -xf CUB_200_2011.tgz
```
#### CUB annotation mat files and pre-computed SfM outputs
Download CUB annotation mat files and pre-computed SfM outputs, from [here](https://drive.google.com/file/d/1Zr4ZN5Hbev2epLn0v2sHYVdyUAJt_dZB/view?usp=sharing) 
This should be saved in ``misc/cachedir`` directory.

