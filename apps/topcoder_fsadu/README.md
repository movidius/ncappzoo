# [Intel Movidius Challenge 3rd Place](https://developer.movidius.com/competition)

Code used to train and export the network is in vast majority from the work of [Technica-Corporation/TF-Movidius-Finetune](https://github.com/Technica-Corporation/TF-Movidius-Finetune)
Basically this is all you need to be able to train and compile a suitable working network for the Intel Movidius NCS, great repository!

Very minor changes were made to this code, for simplicity it's provided here.

The network implementation and Checkpoints used were taken from the MobileNet TensorFlow<sup>TM</sup> slim [models/research/slim/nets/mobilenet_v1.md](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md).

Added to the NCS Challenge dataset, synsets from ImageNet 2011 Fall Release were used as well [ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015.](https://arxiv.org/abs/1409.0575).

## Prerequisites

This code example requires that the following components are available:
1. <a href="https://developer.movidius.com/buy" target="_blank">Movidius Neural Compute Stick</a>
2. <a href="https://developer.movidius.com/start" target="_blank">Movidius Neural Compute SDK</a>
3. <a href="https://github.com/tensorflow/tensorflow/releases/tag/v1.3.0" target="_blank">TensorFlow<sup>TM</sup> 1.3.0</a>

## Results
Refer to the [Problem Statement](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&pm=14775&rd=17058) for details on the scoring process. Scoring Results below were obtained on the provided [80k dataset](https://github.com/movidius/ncappzoo/tree/master/apps/topcoder_example). 

### Fine Tune all layers.

| Model | Accuracy-Top1 |Accuracy-Top5 | Log loss | Inference Time(ms) | Score |
|--------|:------:|:------:|:------:|:------:|:------:|
| MobileNet_v1_1.0_224 |79.94% |95.81% |1.89 |41.52 |908890.88 |
| MobileNet_v1_1.0_192 |78.97% |95.42% |2.04 |30.12 |905281.95 |
| MobileNet_v1_1.0_160 |77.24% |94.61% |2.34 |23.85 |894441.09 |
| MobileNet_v1_0.75_224 |76.29% |94.24% |2.49 |26.97 |886131.23 |
| MobileNet_v1_0.75_192 |74.97% |93.50% |2.75 |20.80 |878126.80 |
| MobileNet_v1_0.75_160 |74.02% |93.06% |2.92 |18.53 |872651.55 |
| MobileNet_v1_0.50_160 |67.50% |89.37% |4.27 |11.91 |824833.36 |


### Fine Tuning only fully connected layers. 

| Model | Accuracy-Top1 |Accuracy-Top5 | Log loss | Inference Time(ms) | Score |
|--------|:------:|:------:|:------:|:------:|:------:|
| MobileNet_v1_1.0_224 |70.09% |91.17% |3.63 |40.52 |825242.58 |  

#  How to train the algorithm

### Prepare data

This is a proposed method to organize the data as shown on step 3, the goal is to expand the dataset provided for the competition with the relevant synsets from ImageNet Fall 11 dataset.

**Note:** If you want to fine tune a model with you own data you can skip this step and continue from step 3.


1. Decompress all the synsets into a training directory. An example on how to do it can be found next; please make sure to update the location where the synsets.tar file is located.
```
python prepare_data/untar_synsets.py
```
 
2. Create a new extended_ground_truth.csv file. Make sure to update the location where Imagenet Synsets directory is located.
```
python prepare_data/extended_ground_truth.py
```

3. Move images under a categories tree, each directory represents a category.
```
prepare_data/move_files_into_categories.ipynb
```
```
root/
                  |->train/
                     |->1/
                     |     |-> 00****1.JPEG
                     |     L-> 00****n.JPEG
                     |->2/
                     |     |-> 00****1.JPEG
                     |     L-> 00****n.JPEG
                     .
                     .
                     .
                     L->200/
                           |-> 00****1.JPEG
                           L-> 00****n.JPEG
```

### Clean the dataset
Find errors where images cannot be read and/or reshaped, resulting in an error while trying to transform them into tfrecords.
    The following notebook can be used to find all the conflicting files, once found you can delete them from the dataset.
```
prepare_data/find_corrupted_files.ipynb
```

### Generate the training and validation tfrecord files

Split the data into training and validation sets by passing the value of the validation_size, once your network is tuned, you can use all the data for training (e.g. --validation_size 0.01).
```
python preprocess_img_dir/create_tfrecord.py --validation_size 0.3
```
### Train the network  

Please refer to the [results tables](https://github.com/saduf/test_readme#results) for a choice.

1.1 Fine Tune all layers (Higher accuracy, longer training time)
```
python train.py --dataset_dir=/home/ubuntu/movidius/train --labels_file=/home/ubuntu/movidius/train/labels.txt --num_epochs 15 --image_size 224 --num_classes 200 --checkpoint_path=./models/checkpoints/mobilenet/01_224/mobilenet_v1_1.0_224.ckpt --checkpoint_exclude_scopes="MobilenetV1/Logits, MobilenetV1/AuxLogits" --log_dir=./tflog/full_run/01_224 --batch_size 16 --preprocessing inception --model_name mobilenet_v1 --tb_logdir=./TB_logdir/full_run/01_224
```

1.2 Fine Tune only fully connected layers (Reduce training time by 50% or more at the expense of accuracy)
```
python train.py --dataset_dir=/home/ubuntu/movidius/train --labels_file=/home/ubuntu/movidius/train/labels.txt --num_epochs 15 --image_size 224 --num_classes 200 --checkpoint_path=./models/checkpoints/mobilenet/01_224/mobilenet_v1_1.0_224.ckpt --checkpoint_exclude_scopes="MobilenetV1/Logits, MobilenetV1/AuxLogits" --log_dir=./tflog/full_run/01_224_FT --batch_size 16 --preprocessing inception --model_name mobilenet_v1 --tb_logdir=./TB_logdir/full_run/01_224_FT --trainable_scopes="MobilenetV1/Logits, MobilenetV1/AuxLogits"
```

2. Monitor your training accuracy and losses in TensorBoard<sup>TM</sup>
```
tensorboard --logdir ./TB_logdir/full_run/01_224
```

### Evaluate the network
```
python eval.py --checkpoint_path ./tflog/full_run/01_224 --num_classes 200 --labels_file /home/ubuntu/movidius/train/labels.txt --dataset_dir /home/ubuntu/movidius/train --file_pattern movidius_%s_*.tfrecord --file_pattern_for_counting movidius --batch_size 16 --preprocessing_name inception --model_name mobilenet_v1 --image_size 224
```

### [Export the network](https://movidius.github.io/ncsdk/tf_compile_guidance.html)
```
python export_inference_graph.py --model_name mobilenet_v1 --image_size 224 --batch_size 1 --num_classes 200 --ckpt_path ./tflog/full_run/01_224/model.ckpt-252435 --output_ckpt_path ./output/full_run/01_224/network
```

### Compiling and Profiling
Transfer your network.meta and weight files to your machine where NCSDK is installed.

1. Compile the network.
```
mvNCCompile network.meta -w network -s 12 -in input -on output -o compiled.graph
```

2. Profilng the network, obtain MFLOPS, bandwidth, and processing time per layer/total.
```
mvNCProfile -in input -on output -s 12 -is 224 224 network.meta
```

### Using the compiled.graph to make inferences on the Intel Movidius NCS. 

1. Make sure to update the path to point to the copiled.graph file.
```
python inference.py path/to/datadir	
```
Inside inference.py make sure to properly point the following variables
```
EXAMPLE_ONLY = False # False for evaluation data, True for test data.
GRAPH_FILE = "../compiled.graph" # Path to the compiled.graph file.
LABELS_FILE = "./labels.txt" # Path to the labels file generated by the tfrecords.
IMGSIZE = "224" # Image size used by the model.
INFERENCE_FILE = "../inferences.csv" # Where the generated inferences file will be located.
```

## TODO
- [ ] Train on different network architectures (e.g. DenseNet). Training and compiling was done, unsuccessfully used for inference. 
- [ ] Support for multi NCS inference (e.g. 3 NCS sticks, inference time/3).
- [ ] Report results. 

## Reference
[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

[Technica-Corporation/TF-Movidius-Finetune](https://github.com/Technica-Corporation/TF-Movidius-Finetune)

[models/research/slim/nets/mobilenet_v1.md](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md).

[Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015.](https://arxiv.org/abs/1409.0575)


## Marathon Match - Solution Description &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![Topcoder](./misc/tcl_re.png "Topcoder")

**1. Introduction:**    
  Tell us a bit about yourself, and why you have decided to participate in the contest.  
  
  * Name: Fernando Sadu
  * Handle: fsadu
  * Placement you achieved in the MM: 3rd.
  * About you: I'm a Computer Technician for the Public Schools in Massachusetts, US. I have a MS. Electrical Engineering from Tecnologico de Monterrey, Mexico. I've worked as a L2 embedded software support engineer in Mobile Communications; I also have experience as a product integrator (build and release), and presales engineer.  
  * Why you participated in the MM: I completed the AIND from Udacity (Dec 2017), with focus on Computer Vision, and when I saw the Intel Movidius NCS challenge, I knew it was the perfect opportunity to put it in practice.

**2. Solution Development**  
  How did you solve the problem? What approaches did you try and what choices did you make, and why? Also, what alternative approaches did you consider?

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;My first approach was to build an Image Classifier using the Keras neural network library, as my experience with Keras was more substantial. I would be able to apply concepts like bottleneck features, batch normalization, l2 regularization, and data augmentation, without much of a learning curve.  
Unfortunately, there was not a clear view on how to export the network.meta file and corresponding weights.  
I was able to build a very shallow CNN with an accuracy of 20%; doing this paved the way to organize the dataset in a directories tree structure, where each directory represents a category.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;I needed a pure TensorFlow<sup>TM</sup> implementation to be able to export the resulting network.meta file. I used a template from my course with Udacity for transfer learning using a VGG network. I was able to achieve ~60% accuracy, a useful point of comparison. In this attempt I also obtained a method to clean the data for I/O errors, and conversion issues on empty images.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Readig the [NCS forums](https://ncsforum.movidius.com/), and reviewing the apps and tutorials provided at	the [NCS App zoo github site](https://github.com/movidius/ncappzoo), I found the [Technica-Corporation/TF-Movidius-Finetune](https://github.com/Technica-Corporation/TF-Movidius-Finetune) gitgub page, which at the same time is an adaptation of the [TF Slim Fine Tuning tools](https://github.com/tensorflow/models/tree/master/research/slim). This repository provides an end-to-end solution to Fine-Tune a TensorFlow<sup>TM</sup> model, and export the resulting network.meta file. Finally I was able to train, evaluate, and export a network.meta file corresponding to a MobileNet_01_224 implementation.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The provided file inferences.py was for a Caffe solution reading images in BGR format instead of the RGB format expected by TensorFlow<sup>TM</sup>. Training features had to be normalized to a range of [0,1], using a mean of 0.5 and a scale factor of 2 to finally bring its values to a range between [-1,1], and finally apply a center crop of 0.875, [see Inception Preprocessing](https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py).  

I decided to go with a MobileNet_v1_1.0_224 Fine-Tuning all the layers, the resulting compiled.graph takes ~40 ms to infer one image, and the compiled.graph file size is around 6.5 MB, compared to the ~200MB for Inception-ResNet-V2. From [ncsdk release notes](https://github.com/movidius/ncsdk/blob/master/docs/release_notes.md), we can see support for Inception-ResNet-V2 wich has a top-1 accuracy ~13% higher according to reported results; on the down side, this is very deep network with more than 10 times the number of parameters than the MobileNet_v1_1.0_224 architecture. Concluding that the Inception-ResNet-V2 model was very expensive to train, and was taking 10 times longer for inference.

**3. Final Approach**  
  Please provide a bulleted description of your final approach. What ideas/decisions/features have been found to be the most important for your solution performance:  

  * Using relevant synsets from [ImageNet Fall 2011 version](http://www.image-net.org) to extend the provided dataset, I saw a jump of ~10% in accuracy.  
  * Generate the extended version of the ground_truth.csv file, around 270,000 images distributed on the 200 categories.  
  * Distribute the dataset into a file system where each directory represents a category. 
  * Clean the data set for I/O errors, and/or reshaping errors. 
  * Define training and validation tfrecords from the training data. Once the network has been tuned and validated, use the complete dataset to train the network.  
  * Fine-Tune all layers for MobileNet_v1_1.0_224 model.  
    * Note: A MobileNet_v1_1.0_160 model cuts the inference time from around 41ms/image to 23ms/image losing only ~ 2% in top-1 accuracy.  
  * Export the network using a batch_size=1 to comply with the NCS specifications. This step will generate the TensorFlow<sup>TM/sup> meta file, as well as the weight files.  
  * Compile the network, the resulting compiled file should be ~6.5MB.  
  * Preprocessing images for inference:  
    * Convert Image to an RGB format.  
    * Normalize features to be in a range of [-1,1], by using a mean of 0.5, and an scale factor of 2.  
  * Finally the inferences.py file returns the key of the labels tfrecords, make sure to map this key to its actual value.

**4. Open Source Resources, Frameworks and Libraries**  
  Please specify the name of the open source resource along with a URL to where it’s housed and its license type:  
  * [Intel NCSDK](https://github.com/movidius/ncsdk), Intel Software Tools License Agreement.  
  * [Technica-Corporation/TF-Movidius-Finetune](https://github.com/Technica-Corporation/TF-Movidius-Finetune), MIT License.  
  * [MobileNet TensorFlow<sup>TM</sup>nsorflow/models/blob/master/research/slim/nets/mobilenet_v1.md), Apache License, v 2.0.  
  * [TensorFlow 1.3.0<sup>TM</sup>](https://github.com/tensorflow/tensorflow), Apache License, v 2.0.  
  * [OpenCV](https://opencv.org/), 3-clause BSD License.  
  * [Numpy](https://github.com/numpy/numpy), Copyright (c) 2005-2017, NumPy Developers.
  * [ImageNet Fall 2011](http://www.image-net.org), ImageNet project and training data includes internet photos that may be subject to copyright.
  
  
**5. Potential Algorithm Improvements**  
  Please specify any potential improvements that can be made to the algorithm:  
  
  * Try DenseNet 121 (k=32) which has a memory footprint of ~ 33 MB compared to the  ~17 MB of the MobileNet_01_224 model, aproximately double the number of trainable parameters, it could represent a gain of ~8% in top-1 accuracy, [see Keras documentations for individual models](https://keras.io/applications/).  
  * Implement multi NCS support for 3 NCS sticks, inference time/3, this should contribute to the overall Logarithmic Loss by decreasing the inference time.
  * Report results and compare ConvNets scores, visualize tradeoff scenarios for accuracy, speed, and memory footprint.

**6. Algorithm Limitations**  
  Please specify any potential limitations with the algorithm:  
  
  * Limited to MobileNet model, please see results [here](#mobilenet-accuracy-test-results---80k-dataset---fine-tune-all-layers). 
 * Only Inception preprocessing was implemented at the inference.py supporting file, other preprocessing implementations are needed to support different network architectures, e.g. vgg preprocessing.

**7. Deployment Guide**  
  Please provide the exact steps required to build and deploy the code: [Do this](#compiling-and-profiling)

**8. Final Verification**  
  Please provide instructions that explain how to train the algorithm and have it execute against sample data: [Do this](#how-to-train-the-algorithm)
