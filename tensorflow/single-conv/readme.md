How to run this app:

```
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/ashwinvijayakumar/ncappzoo
cd ~/workspace/ncappzoo/tensorflow/single-conv/data
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
cd  ~/workspace/ncappzoo/tensorflow/single-conv
# Make sure you have activated TensorFlow virtual environment (if required)
python3 single-conv-train.py
python3 single-conv-inference.py
cd ~/workspace/ncappzoo/tensorflow/single-conv/model
mvNCCompile -s 12 model-inference.meta
```

You can now use ~/workspace/ncappzoo/image-classifier or rapid-image-classifier to load the graph file and run inference.

> The default image-classifier & rapid-image-classifier will need to be updated so it is pointing to the right graph and test data.
