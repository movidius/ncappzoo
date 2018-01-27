.PHONY: all
all: data train export

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
	(python3 single-conv-train.py)

.PHONY: export
export:
	@echo "\nExporting network model for inference..." 
	(python3 single-conv-inference.py)

.PHONY: compile
compile:
	@echo "\nCompiling model to Movidius graph..."
	(cd model; mvNCCompile -s 12 model-inference.meta;)

.PHONY: profile
profile:
	@echo "\nProfiling the model..."
	(cd model; mvNCProfile -s 12 model-inference.meta;)

.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo "  make help - Shows this message.";
	@echo "  make - Builds all dependencies, but does not run this program.";
	@echo "  make data - Downloads dataset.;"
	@echo "  make train - Start training the neural network.";
	@echo "  make export - Export the trained model for inference.";
	@echo "  make compile - Convert the trained model into Movidius graph file.";
	@echo "  make profile - Run the model on NCS and extract complexity, bandwidth and execution time for each layer.";

.PHONY: clean
clean:
	@echo "\nMaking clean...";
	rm -rf model
