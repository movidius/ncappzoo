
MODEL_CKPT=model
OUTPUT_NODE_NAME=dense2/dense/Softmax

YELLOW='\033[1;33m'
NOCOLOR='\033[0m'

.PHONY: all
all: check_tf data train profile

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
	(python3 single-conv-train.py --data_dir=data --model_dir=model --model_name=$(MODEL_CKPT))

check_tf:
ifndef TF_SRC_PATH
	$(error TF_SRC_PATH is not defined. Run `export TF_SRC_PATH=path/to/your/tensorflow/source/repo`)
endif

.PHONY: freeze
freeze: check_tf
	@echo "\nFreezing model for inference..." 
	(cd model; python3 $(TF_SRC_PATH)/tensorflow/python/tools/freeze_graph.py \
             --input_graph=$(MODEL_CKPT).pb \
             --input_binary=true \
             --input_checkpoint=$(MODEL_CKPT).ckpt \
             --output_graph=$(MODEL_CKPT)_frozen.pb \
             --output_node_name=$(OUTPUT_NODE_NAME);)

.PHONY: compile
compile: freeze
	@echo "\nCompiling model to Movidius graph..."
	(cd model; mvNCCompile -s 12 $(MODEL_CKPT)_frozen.pb -in=input -on=$(OUTPUT_NODE_NAME);)
	@echo $(YELLOW)"Movidius graph generated! You can run inferences by using ncappzoo/apps/image-classifier project.\n"$(NOCOLOR)

.PHONY: profile
profile: freeze
	@echo "\nProfiling the model..."
	(cd model; mvNCProfile -s 12 $(MODEL_CKPT)_frozen.pb -in=input -on=$(OUTPUT_NODE_NAME);)
	@echo $(YELLOW)"Movidius graph generated! You can run inferences by using ncappzoo/apps/image-classifier project.\n"$(NOCOLOR)

.PHONY: check
check: freeze
	@echo "\nComparing results with standard TensorFlow..."
	(cd model; mvNCCheck -s 12 $(MODEL_CKPT)_frozen.pb -in=input -on=$(OUTPUT_NODE_NAME);)
	@echo $(YELLOW)"Movidius graph generated! You can run inferences by using ncappzoo/apps/image-classifier project.\n"$(NOCOLOR)

.PHONY: run
run: compile
	@echo "\nRunning inferences using image-classifier project..."
	(python3 ../../apps/image-classifier/image-classifier.py --graph ./model/graph --dim 28 28 --mean 0 --scale 0.00392 --colormode "monochrome" --labels ./data/categories.txt --image ./data/testing/5/3698.png)

.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo "  make help - Shows this message.";
	@echo "  make - Builds all dependencies, but does not run the inference app.";
	@echo "  make data - Downloads dataset.;"
	@echo "  make train - Start training the neural network.";
	@echo "  make freeze - Export the trained model for inference.";
	@echo "  make compile - Convert the trained model into Movidius graph file.";
	@echo "  make check - Compare inference results with that of TensorFlow running on CPU/GPU.";
	@echo "  make profile - Run the model on NCS and extract complexity, bandwidth and execution time for each layer.";

.PHONY: clean
clean:
	@echo "\nMaking clean...";
	rm -rf model
