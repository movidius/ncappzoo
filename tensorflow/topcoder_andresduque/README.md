# [Intel Movidius Challenge 5th Place](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17058&pm=14775)

My model is created finetuning TensorFlow Inception-V3 (slim version) pretrained model and exporting it to a format that can be compiled using [Movidius NCSDK](https://github.com/movidius/ncsdk).  
Code in [TF-Movidius-Finetune](https://github.com/Technica-Corporation/TF-Movidius-Finetune) repository is used to train and export the model.

Exporting a TensorFlow compatible model for Movidius is still experimental and there are some challenges, 
the authors of this repository have some code and guidelines to streamline the process of finetuning from a pretrained TensorFlow model.

## Organization
* `supporting/*` Contains all required files to train, compile and test the network.

## Third party used resources
* Code in [TF-Movidius-Finetune](https://github.com/Technica-Corporation/TF-Movidius-Finetune)
* [Tensorflow Inception-V3](http://download.tensorflow.org/models/inceptionv320160828.tar.gz) pretrained model
* [Training data](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17058&pm=14775) provided by **TopCoder**
* Modified [example code](https://github.com/movidius/ncappzoo/tree/master/apps/topcoder_example) provided by **TopCoder**

## Requirements
### Training

* I used Amazon AWS for training. The used AMI is Deep Learning AMI with Source Code Ubuntu v4.0 (ami-0bd0e56e), but similar setups should also work
* I trained using Amazon AWS p3.2xlarge instance
* Tensorflow 1.5.0
* Python 2.7

### Testing / Inference
* [Movidius NCSDK](https://github.com/movidius/ncsdk)
* [Movidius USB Stick](https://developer.movidius.com)
* Ubuntu 16.04
* Python 3


## Training

Make a directory in a disk with enough space, e.g. `/movidius/`, this will be the root folder where all scripts should be run

Copy all files in `/supporting/` dir to `/movidius/supporting/`

All commands assume we are in root directory (e.g. `/movidius/`)

Clone TF-Movidius-Finetune, some scripts in this repository are used for training and exporting a Movidius NCSDK compatible model.

```
git clone https://github.com/Technica-Corporation/TF-Movidius-Finetune
```

TF-Movidius-Finetune `create_tfrecord.py` file was modified to save the validation images list in CSV. 
This file can be used to validate Movidius inference results. (see *Make inferences*)

```
sudo cp ./supporting/create_tfrecord.py ./TF-Movidius-Finetune/preprocess_img_dir/create_tfrecord.py
```

Set folders permissions to avoid errors

```
sudo chown -R ubuntu ./*
```

Download TensorFlow pretrained Inception-V3 model

```
mkdir -p ./data/
wget -nc -O ./data/inception_v3_2016_08_28.tar.gz "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
tar xzf ./data/inception_v3_2016_08_28.tar.gz -C ./data/
```

Download training data, for simplicity it's assumed paths will be `./data/training/*` and `./data/training_ground_truth.csv` (If training data is in other folder set the correct path arguments for all scripts)

```
mkdir -p ./data/
wget -nc -O ./data/training.tar "https://www.topcoder.com/contest/problem/IntelMovidius/training.tar"
wget -nc -O ./data/training_ground_truth.csv "https://www.topcoder.com/contest/problem/IntelMovidius/training_ground_truth.csv"
sudo tar xf ./data/training.tar
```

Initialize training images directory, this will copy and organize all images using a custom folder structure compatible with the training scripts (see `init_images_dir.py` file for details)

```
sudo python ./supporting/init_images_dir.py -source-dir ./data/training -dest-dir ./data/training_folders -images-file ./data/training_ground_truth.csv
```
Some corrupt training images found are deleted to avoid errors.
```
sudo rm ./data/training_folders/133/training_37021.jpg
```

Images are converted to tfrecord files (used by TensorFlow in training process).
`-validation_size` is used to define the percent of images that will be used for validation.

```
sudo chown -R ubuntu ./*
python ./TF-Movidius-Finetune/preprocess_img_dir/create_tfrecord.py --dataset_dir=./data/training_folders --tfrecord_filename=movidius_dataset --random_seed 65845 --validation_size 0.025
```

We can now start finetuning the model.  
Process can take some hours, will vary depending on hardware. Console output will help estimate how long it will take.
Using AWS p3.2xlarge training time is approximately 3 hours.
Training script will save checkpoints periodically. Training process can be stopped at any time and resumed using the same command.

```
python TF-Movidius-Finetune/train.py --model_name inception_v3 --learning_rate 0.00005 --dataset_dir=./data/training_folders/ --labels_file=./data/training_folders/labels.txt --num_epochs 15 --image_size 299 --checkpoint_path=./data/inception_v3.ckpt --checkpoint_exclude_scopes="InceptionV3/Logits,InceptionV3/AuxLogits" --log_dir=./logs/ --batch_size 64 --preprocessing inception --file_pattern movidius_dataset_%s_*.tfrecord --file_pattern_for_counting movidius_dataset --num_classes 200
```

We can follow the next steps even if training is stopped midway (all scripts will always use the latest checkpoint), but in this case the resulting model will have less accuracy.

*Note:* Trained model can vary each time training is done. This is expected in TensorFlow. Accuracy of different models should be very similar.

If for some reason we want to train a new model from start, we need to remove the checkpoint files log folder, otherwise training process will resume from the last saved checkpoint.
```
sudo rm -rf ./logs
```

After finetuning, we can optionally evaluate the model with the validation set

```
python TF-Movidius-Finetune/eval.py --model_name inception_v3 --checkpoint_path ./logs/ --num_classes 200 --labels_file=./data/training_folders/labels.txt --dataset_dir=./data/training_folders/ --file_pattern movidius_dataset_%s_*.tfrecord --file_pattern_for_counting movidius_dataset --batch_size 2
```

To use the model in Movidius device, we are required to export it to a supported format recognized by Movidius NCSDK. 
This step will create the following files (in `./output/` directory)

* network.meta
* network.data-00000-of-00001
* network.index

These files can be used to compile the model in Movidius NCSDK. (See *Compile graph* step)

* `--ckpt_path` is the input folder where checkpoints are located (this command will use a helper script to find the latest trained checkpoint)
* `--output_ckpt_path` is the output path to save the exported files

```
python ./TF-Movidius-Finetune/export_inference_graph.py --model_name inception_v3 --image_size 299 --ckpt_path "$(sudo bash ./supporting/latest_checkpoint.sh ./logs2bk)" --output_ckpt_path ./output/network --num_classes 200 --batch_size 1
```

We can use the exported files to compile a Movidius NCSDK graph file. (See *Compile graph* step)



## Compile graph

Be sure Movidius NCSDK is correctly [installed](https://movidius.github.io/ncsdk/install.html)  
To compile graph, the following files are required (output of *Training* step).

* network.meta
* network.data-00000-of-00001
* network.index

Make a new directory in a disk with enough space, e.g. `/movidius/`, this will be the root folder where all scripts should be run.

Copy all files in `/supporting/` dir to `/movidius/supporting/`

All commands assume we are in root directory (e.g. `/movidius/`)

To compile graph, use following command

```
mvNCCompile network.meta -in=input -on=final_result -o compiled.graph -s12
```

Output file `compiled.graph` should be created.

## Make inferences

This step assume current directory is the same one created at *Compile graph* step.

* `-images-dir` Should point to the folder containing the images to infer (**.jpg** extension)
* `-output-file` Output CSV file to save inferences
* `-graph-file` The compiled graph file path
* `-labels-map-file` At inference, this file is used to map predicted tensorflow label indexes to label names (ground truth labels). (This file is created by tensorflow at training step)
* `-images-file` Optionally we can pass an image list CSV file (same format as competition ground truth). 
               In this case only the images in the CSV file will be processed (found in `-images-dir`). 
               Also the results will be scored and results output in console.

Following command will infer all images in `./data/provisional` folder

```
sudo python3 ./supporting/inferences.py -images-dir ./data/provisional -output-file ./inferences.csv -graph-file ./compiled.graph -labels-map-file ./supporting/labels.txt
```

Following command will validate trained model, using validation set (see *Training* step)

```
sudo python3 ./supporting/inferences.py -images-dir ./data/training -images-file ./data/validate.csv -output-file ./inferences.csv -graph-file ./compiled.graph -labels-map-file ./supporting/labels.txt
```

# Marathon Match Solution Description

## Introduction
Tell us a bit about yourself, and why you have decided to participate in the contest.

**Name:** Andres Duque

**Handle:** andresduque

**Placement you achieved in the MM:** 5th

**Why you participated in the MM:** To learn more about deep learning. Buying Movidius USB stick was required to compete in contest but I found device interesting (small and low power consumption).

## Solution Development 
I started competition using **Caffe**, then switched to **TensorFlow** to test more recent and accurate networks. I used finetuning to train all the networks. I tested the following networks (in order)

**Caffe - Alexnet:** Worked fine but the accuracy was inferior to other networks.

**Caffe - Googlenet:** Better accuracy than Alexnet, with almost equal inference time. Best provisional result using this was **813302.87**

**TensorFlow - Inception V3:** Accuracy is better than Googlenet. Training time is increased and inference time is approximately 3x than Googlenet. My last submission provisional score using this network is **855131.13**

## Final Approach
Finetune [Inception-V3](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz) TensorFlow slim model

To train, I used the code in github [TF-Movidius-Finetune](https://github.com/Technica-Corporation/TF-Movidius-Finetune) repository 

## Training

I used only the provided training data for contest.
By default  [TF-Movidius-Finetune](https://github.com/Technica-Corporation/TF-Movidius-Finetune) use some data augmentation techniques like random crop, random flip and and random distortion of colors. I used the following training parameters (Other parameters where left unchanged from original code):

* Train all layers (don't freeze)
* Batch size: 64
* Learning rate: 0.00005
* Epochs: 15

## Inference

Images are preprocessed in the same way as in training phase (Inception-V3 preprocessing)

* Each pixel value will be (for each channel): (value - 128) * (1/128)
* Image is resized to **299x299** pixels. Because images are not square in shape, when resizing we can cause a  slightly distortion. I found a small improvement in score using a technique used in Inception **V3 Movidus NCSDK** examples where images are first cropped to square shape and then resized. (File *inferences.py*, *load_BGR()* function)
 
## Open Source Resources, Frameworks and Libraries

#### TF-Movidius-Finetune
* https://github.com/Technica-Corporation/TF-Movidius-Finetune
* MIT License
#### Deep Learning AMI with Source Code Ubuntu v4.0
* Ami-0bd0e56e
* Movidius NCSDK
* https://github.com/movidius/ncsdk
#### Provided training data and resources
* https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17058&pm=14775
* https://github.com/movidius/ncappzoo/tree/master/apps/topcoder_example

## Potential Algorithm Improvements
* Try other data augmentation techniques.
* Use more training data.
* Training parameters could be better tuned. (Learning rate and decay, used optimizer, etc)

## Algorithm Limitations
* More inference and training time than other networks.

## Contest feedback
* Problem statement was clear
* Provided training data was good, but some images are corrupt.
* Scoring was clear. I think if inference time was penalized more heavily contest could be more interesting. I found the best scores using the more heavy network (**Inception-V3**), this give very good accuracy but I think the Intel Movidius device is optimized for low power consumption and fast inference (e.g. real time applications).