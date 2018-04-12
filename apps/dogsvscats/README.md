# Dogs vs Cats on Intel® Movidius™ Neural Compute Stick (NCS)

An advanced-level project demonstrating the process of training a customized version of GoogLeNet model using Caffe, and then deploy it to Intel® Movidius™ Neural Compute Stick (NCS) for inference. You can read more about this project (and a step-by-step guide) on <a href="https://movidius.github.io/blog/deploying-custom-caffe-models/">NCS developer blog</a>. 

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

3. Now run the below steps on your training hardware, to train your model. This step might take anywhere from a couple minutes to hours, even days depending on how powerful you training hardware is.

   ~~~
   cd ~/workspace/ncappzoo/apps/dogsvscats
   export CAFFE_PATH=/PATH/TO/YOUR/CAFFE/INSTALL_DIR
   make train
   ~~~

   If everything went well, you should see an output similar to this:

   ~~~
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
   Training the model...
   ...
   I0411 20:19:02.791424  9236 solver.cpp:272] Solving GoogleNet
   I0411 20:19:02.791426  9236 solver.cpp:273] Learning Rate Policy: step
   I0411 20:19:03.141243  9236 solver.cpp:218] Iteration 0 (0 iter/s, 0.34977s/40 iters), loss = 2.4185
   I0411 20:19:03.141291  9236 solver.cpp:237]     Train net output #0: loss1/loss1 = 4.40975 (* 0.3 = 1.32293 loss)
   I0411 20:19:03.141300  9236 solver.cpp:237]     Train net output #1: loss2/loss2 = 1.00351 (* 0.3 = 0.301052 loss)
   I0411 20:19:03.141307  9236 solver.cpp:237]     Train net output #2: loss3/loss3 = 0.794524 (* 1 = 0.794524 loss)
   I0411 20:19:03.141319  9236 sgd_solver.cpp:105] Iteration 0, lr = 0.01
   I0411 20:19:14.655712  9236 solver.cpp:218] Iteration 40 (3.47398 iter/s, 11.5142s/40 iters), loss = 1.68719
   I0411 20:19:14.655757  9236 solver.cpp:237]     Train net output #0: loss1/loss1 = 1.04423 (* 0.3 = 0.313268 loss)
   I0411 20:19:14.655766  9236 solver.cpp:237]     Train net output #1: loss2/loss2 = 0.641296 (* 0.3 = 0.192389 loss)
   I0411 20:19:14.655772  9236 solver.cpp:237]     Train net output #2: loss3/loss3 = 0.706645 (* 1 = 0.706645 loss)
   I0411 20:19:14.655778  9236 sgd_solver.cpp:105] Iteration 40, lr = 0.01
   ...
   ~~~

4. Assuming that the training process went without a hitch, and `bvlc_googlenet/org/bvlc_googlenet_iter_40000.caffemodel` exists, run `make` or `make profile` to convert the `.caffemodel` file into a Movidius Graph file. You should see the following output:

   ~~~
   make

   Profiling the model...
   (cd bvlc_googlenet/org; mvNCProfile -s 12 deploy.prototxt -w bvlc_googlenet_iter_40000.caffemodel;)
   ...
   70   inception_5b/5x5                                  15.1   452.8   0.908
   71   inception_5b/pool                                  0.4   363.2   0.214
   72   inception_5b/pool_proj                            10.4   498.1   0.564
   73   pool5/7x7_s1                                       0.1   462.8   0.207
   74   loss3/classifier_dc                                0.0   128.6   0.046
   75   prob                                               0.0     0.1   0.049
   ---------------------------------------------------------------------------
                                  Total inference time                   94.71
   ---------------------------------------------------------------------------
   ~~~

5. Now that you have a Movidius graph file for your custom network, you can use `image-classifier`, `live-image-classifier` or `rapid-image-classifer` app to run inferences on the Neural Compute Stick.

   ~~~
   make run
   ...
   Predicted 10171.jpg as dog in 73.69 ms with 79.0% confidence.
   Predicted 10172.jpg as cat in 73.19 ms with 76.3% confidence.
   Predicted 10173.jpg as dog in 73.49 ms with 57.7% confidence.
   Predicted 10174.jpg as dog in 73.37 ms with 56.0% confidence.
   Predicted 10175.jpg as dog in 72.98 ms with 77.2% confidence.
   Predicted 10176.jpg as cat in 73.22 ms with 84.7% confidence.
   Predicted 10177.jpg as cat in 73.20 ms with 69.2% confidence.
   ==============================================================
   ~~~

6. Head over to the <a href="https://movidius.github.io/blog/deploying-custom-caffe-models/">NCS developer blog</a> for more tips n tricks on customizing this project.

## Troubleshooting

1. If you see the below error, make sure you have exported your caffe installation path to `CAFFE_PATH`. See step 3 in `Run this example` section.

   ~~~
   ./create-lmdb.sh: 45: ./create-lmdb.sh: /build/tools//convert_imageset: not found
   ~~~

2. If you see the below error, make sure you have trained the model and `bvlc_googlenet/org/bvlc_googlenet_iter_40000.caffemodel` exists. You can train the model by running `make train`.

   ~~~
   [Error 9] Argument Error: Network weight cannot be found.
   ~~~

3. If you see the below error while running `image-classifier`, make sure you have changed `NUM_PREDICTIONS = 5` to `NUM_PREDICTIONS = 2` in `apps/image-classifier/image-classifier.py`

   ~~~
   IndexError: index 2 is out of bounds for axis 0 with size 2
   ~~~
