
# Introduction
The [Inception ResNet V2] network can be used for image classification.
To be able using it do the following:

## Prerequisites

This code example requires that the following components are available:
1. <a href="https://developer.movidius.com/buy" target="_blank">Movidius Neural Compute Stick</a>
2. <a href="https://developer.movidius.com/start" target="_blank">Movidius Neural Compute SDK</a>
3. <a href="https://github.com/tensorflow/tensorflow" target="_blank">TensorFlow source repo</a>
4. <a href="https://github.com/tensorflow/models" target="_blank">TensorFlow models repo</a>

## Running this example

~~~
mkdir -p ~/workspace/tensorflow

# Clone TensorFlow source and models repo
cd ~/workspace/tensorflow
git clone https://github.com/tensorflow/tensorflow
git clone https://github.com/tensorflow/models

# Clone NC App Zoo
cd ~/workspace
git clone https://github.com/movidius/ncappzoo

# Download, export, freeze and profile model
~~~

1. Download the TensorFlow checkpoint file using
wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz && \
tar zxf inception_resnet_v2_2016_08_30.tar.gz && \
rm inception_resnet_v2_2016_08_30.tar.gz
Now you will have the pretrained model inception_resnet_v___ file.
	
2. Export GraphDef file using
python3 ~/workspace/tensorflow/models/research/slim/export_inference_graph.py \
--alsologtostderr --model_name=inception_resnet_v2 --batch_size=1 \
--dataset_name=imagenet --image_size=299 --output_file=inception_resnet_v2.pb


3. Freeze the model for inference using
python3 ~/workspace/tensorflow/tensorflow/python/tools/freeze_graph.py \
--input_graph=inception_resnet_v2.pb --input_binary=true \
--input_checkpoint=inception_resnet_v2_2016_08_30.ckpt \
--output_graph=inception_resnet_v2_frozen.pb \
--output_node_name=InceptionResnetV2/Logits/Predictions


4. Compile, Profile and Check the network using the Neural Compute SDK
mvNCCompile -s 12 inception_resnet_v2_frozen.pb -in=input -on=InceptionResnetV2/Logits/Predictions
mvNCProfile -s 12 inception_resnet_v2_frozen.pb -in=input -on=InceptionResnetV2/Logits/Predictions

There is a run.py provided that does a single inference on a provided image as an example on
how to use the network using the Neural Compute API

