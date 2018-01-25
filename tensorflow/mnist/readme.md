How to run this app:

```
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/ashwinvijayakumar/ncappzoo
cd  ~/workspace/ncappzoo/tensorflow/mnist
# Make sure you have activated TensorFlow virtual environment (if required)
python3 mnist_deep.py
python3 mnist_inference.py
cd ~/workspace/ncappzoo/tensorflow/mnist/model
mvNCCompile -s 12 model-inference.meta
```

You can now use ~/workspace/ncappzoo/image-classifier or rapid-image-classifier to load the graph file and run inference.

> The default image-classifier & rapid-image-classifier will need to be updated
so it is pointing to the right graph and test data.

