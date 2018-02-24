
VERSION?=v1
DEPTH?=1.0
IMGSIZE?=224
MODEL_CKPT=mobilenet_$(VERSION)_$(DEPTH)_$(IMGSIZE)
MODEL_CKPT_TAR=$(MODEL_CKPT)_2017_06_14.tar.gz

# MobileNet model names as defined in https://github.com/tensorflow/models/blob/master/research/slim/nets/nets_factory.py
ifeq ($(DEPTH),1.0) 
MODEL_NAME=mobilenet_$(VERSION)
else ifeq ($(DEPTH),0.75)
MODEL_NAME=mobilenet_$(VERSION)_075
else ifeq ($(DEPTH),0.50)
MODEL_NAME=mobilenet_$(VERSION)_050
else ifeq ($(DEPTH),0.25)
MODEL_NAME=mobilenet_$(VERSION)_025
endif

OUTPUT_NODE_NAME=MobilenetV1/Predictions/Reshape_1

YELLOW='\033[1;33m'
NOCOLOR='\033[0m'

.PHONY: all
all: profile

.PHONY: checkpoint
checkpoint:
	@echo "\nDownloading checkpoint files..."
	(mkdir -p model)
	(cd model; wget -N http://download.tensorflow.org/models/$(MODEL_CKPT_TAR);)
	(cd model; tar -xvf $(MODEL_CKPT_TAR);)

check_model:
ifndef TF_MODELS_PATH
	$(error TF_MODELS_PATH is not defined. Run `export TF_MODELS_PATH=path/to/your/tensorflow/models/repo`)
endif

.PHONY: export
export: check_model
	@echo "\nExporting model for inference..."
	(cd model; python3 $(TF_MODELS_PATH)/research/slim/export_inference_graph.py \
             --alsologtostderr \
             --model_name=$(MODEL_NAME) \
             --batch_size=1 \
             --image_size=$(IMGSIZE) \
             --output_file=$(MODEL_CKPT).pb;)

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
compile: checkpoint export freeze
	@echo "\nCompiling model to Movidius graph..."
	(cd model; mvNCCompile -s 12 $(MODEL_CKPT)_frozen.pb -in=input -on=$(OUTPUT_NODE_NAME);)
	@echo $(YELLOW)"Movidius graph generated! You can run inferences using ncappzoo/apps/image-classifier project.\n"$(NOCOLOR)

.PHONY: profile
profile: checkpoint export freeze
	@echo "\nProfiling the model..."
	(cd model; mvNCProfile -s 12 $(MODEL_CKPT)_frozen.pb -in=input -on=$(OUTPUT_NODE_NAME);)
	@echo $(YELLOW)"Movidius graph generated! You can run inferences using ncappzoo/apps/image-classifier project.\n"$(NOCOLOR)

.PHONY: check
check: checkpoint export freeze
	@echo "\nComparing results with standard Tensorflow..."
	(cd model; mvNCCheck -s 12 $(MODEL_CKPT)_frozen.pb -in=input -on=$(OUTPUT_NODE_NAME);)
	@echo $(YELLOW)"Movidius graph generated! You can run inferences using ncappzoo/apps/image-classifier project.\n"$(NOCOLOR)

.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo "  make help - Shows this message.";
	@echo "  make - Builds all dependencies, but does not run this program.";
	@echo "  make checkpoint - Downloads pre-trained checkpoint files.;"
	@echo "  make export - Export the neural network model for inference.";
	@echo "  make freeze - Freeze the neural network model for inference.";
	@echo "  make compile - Convert the frozen model into Movidius graph file.";
	@echo "  make check - Compare inference results with that of TensorFlow running on CPU/GPU.";
	@echo "  make profile - Run the model on NCS and extract complexity, bandwidth and execution time for each layer.";

.PHONY: clean
clean:
	@echo "\nMaking clean...";
	rm -rf model
