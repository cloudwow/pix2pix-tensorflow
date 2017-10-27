# pix2pix-tensorflow

Based on [pix2pix](https://phillipi.github.io/pix2pix/) by Isola et al.

[Article about this implemention](https://affinelayer.com/pix2pix/)

Tensorflow implementation of pix2pix.  Learns a mapping from input images to output images, like these examples from the original paper:

<img src="docs/examples.jpg" width="900px"/>

## Setup

### Prerequisites
- Tensorflow 1.0.0

### Recommended
- Linux with Tensorflow GPU edition + cuDNN

### Getting Started


# clone this repo
git clone https://github.com/cloudwow/pix2pix-tensorflow.git
cd pix2pix-tensorflow
# download the CMP Facades dataset (generated from http://cmp.felk.cvut.cz/~tylecr1/facade/)
python tools/download_southpark.py southpark
# train the model (this may take 1-8 hours depending on GPU, on CPU you will be waiting for a bit)
./train_local.sh 
