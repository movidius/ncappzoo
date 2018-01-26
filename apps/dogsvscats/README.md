# Dogs vs Cats on Intel Movidius™ Neural Compute Stick (NCS)

Dataset preparation script for Kaggle's Dog vs Cats competition. This project was created to support a tutorial on how to train a customized version of GoogLeNet using Caffe, and then deploy it to Intel® Movidius™ Neural Compute Stick (NCS) for inference. You can read more about this project (and a step-by-step guide) on <a href="https://movidius.github.io/blog/deploying-custom-caffe-models/">NCS developer blog</a>. 

## Prerequisites

This code example requires that the following components are available:
1. <a href="https://developer.movidius.com/buy" target="_blank">Movidius Neural Compute Stick</a>
2. <a href="https://developer.movidius.com/start" target="_blank">Movidius Neural Compute SDK</a>
3. A computer that is powerfull enough to run network training. It could be a GPU or a CPU-Only system.

## Running the Example

1. Clone NC App Zoo to your training system.

   ~~~
   mkdir -p ~/workspace
   cd ~/workspace
   git clone https://github.com/movidius/ncappzoo
   ~~~

2. Download <a href="https://www.kaggle.com/c/dogs-vs-cats/data">test1.zip and train1.zip</a> from Kaggle, into `~/workspace/ncappzoo/apps/dogsvscats/data`.

3. Now run the below steps on your training hardware:

   ~~~
   cd ~/workspace/ncappzoo/apps/dogsvscats
   export CAFFE_PATH=/PATH/TO/YOUR/CAFFE/INSTALL_DIR
   make run
   ~~~

   If everything went well, you should see an output similar to this:

   ~~~
   Installing dependencies...
   ...
   Extracting dataset...
   ...
   Creating labels file...
   ...
   Creating train lmdb...
   ...
   Creating val lmdb...
   ...
   Computing image mean...
   ...
   I0125 17:11:38.155160  6739 compute_image_mean.cpp:108] Write to data/dogsvscats-train-mean.binaryproto
   I0125 17:11:38.156217  6739 compute_image_mean.cpp:114] Number of channels: 3
   I0125 17:11:38.156314  6739 compute_image_mean.cpp:119] mean_value channel [0]: 106.202
   I0125 17:11:38.156432  6739 compute_image_mean.cpp:119] mean_value channel [1]: 115.912
   I0125 17:11:38.156528  6739 compute_image_mean.cpp:119] mean_value channel [2]: 124.449
   I0125 17:11:38.604861  6748 db_lmdb.cpp:35] Opened lmdb data/dogsvscats_val_lmdb
   I0125 17:11:38.606081  6748 compute_image_mean.cpp:70] Starting iteration
   I0125 17:11:38.868082  6748 compute_image_mean.cpp:101] Processed 4167 files.
   I0125 17:11:38.869017  6748 compute_image_mean.cpp:108] Write to data/dogsvscats-val-mean.binaryproto
   I0125 17:11:38.870085  6748 compute_image_mean.cpp:114] Number of channels: 3
   I0125 17:11:38.870182  6748 compute_image_mean.cpp:119] mean_value channel [0]: 106.232
   I0125 17:11:38.870297  6748 compute_image_mean.cpp:119] mean_value channel [1]: 116.013
   I0125 17:11:38.870393  6748 compute_image_mean.cpp:119] mean_value channel [2]: 124.181
   ~~~

4. Head over to the <a href="https://movidius.github.io/blog/deploying-custom-caffe-models/">NCS developer blog</a> for next steps.

## Troubleshooting

1. If you see the below error, make sure you have exported your caffe installation path to `CAFFE_PATH`. See step 3 in `Run this example` section.

   ~~~
   ./create-lmdb.sh: 45: ./create-lmdb.sh: /build/tools//convert_imageset: not found
   ~~~
