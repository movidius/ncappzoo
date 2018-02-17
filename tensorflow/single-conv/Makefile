.PHONY: all
all: data train freeze

.PHONY: data
data:
	@echo "\nDownloading dataset..."
	@(cd data; wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz;)
	@(cd data; wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz;)
	@(cd data; wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz;)
	@(cd data; wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz;)

.PHONY: train
train:
	@echo "\nTraining Fashion MNIST..."
	(python3 single-conv-train.py --model_dir=model)

check_tf:
ifndef TF_SRC_PATH
	$(error TF_SRC_PATH is not defined. Run `export TF_SRC_PATH=path/to/your/tensorflow/source/repo`)
endif

.PHONY: freeze
freeze: check_tf
	@echo "\nExporting network model for inference..." 
	(cd model; python3 $(TF_SRC_PATH)/tensorflow/python/tools/freeze_graph.py \
             --input_graph=model.pbtxt \
             --input_checkpoint=model.ckpt \
             --output_graph=model-frozen.pb \
             --output_node_name=dense2/dense/Softmax;)

.PHONY: compile
compile:
	@echo "\nCompiling model to Movidius graph..."
	(cd model; mvNCCompile -s 12 model-frozen.pb -in=input -on=dense2/dense/Softmax;)

.PHONY: profile
profile:
	@echo "\nProfiling the model..."
	(cd model; mvNCProfile -s 12 model-frozen.pb -in=input -on=dense2/dense/Softmax;)

.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo "  make help - Shows this message.";
	@echo "  make - Builds all dependencies, but does not run this program.";
	@echo "  make data - Downloads dataset.;"
	@echo "  make train - Start training the neural network.";
	@echo "  make freeze - Export the trained model for inference.";
	@echo "  make compile - Convert the trained model into Movidius graph file.";
	@echo "  make profile - Run the model on NCS and extract complexity, bandwidth and execution time for each layer.";

.PHONY: clean
clean:
	@echo "\nMaking clean...";
	rm -rf model
