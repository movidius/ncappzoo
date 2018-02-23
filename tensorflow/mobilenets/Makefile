
# MobileNet pre-trained checkpoint files listed @ https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md
MOBILENET_DEPTH?=1.0
MOBILENET_SIZE?=224
MOBILENET_CKPT=mobilenet_v1_$(MOBILENET_DEPTH)_$(MOBILENET_SIZE)
MOBILENET_CKPT_TAR=$(MOBILENET_CKPT)_2017_06_14.tar.gz

# MobileNet model names as defined in https://github.com/tensorflow/models/blob/master/research/slim/nets/nets_factory.py
ifeq ($(MOBILENET_DEPTH),1.0) 
MOBILENET_NAME=mobilenet_v1
else ifeq ($(MOBILENET_DEPTH),0.75)
MOBILENET_NAME=mobilenet_v1_075
else ifeq ($(MOBILENET_DEPTH),0.50)
MOBILENET_NAME=mobilenet_v1_050
else ifeq ($(MOBILENET_DEPTH),0.25)
MOBILENET_NAME=mobilenet_v1_025
endif

.PHONY: all
all: checkpoint export freeze profile

.PHONY: checkpoint
checkpoint:
	@echo "\nDownloading checkpoint files..."
	(mkdir -p model)
	(cd model; wget -N http://download.tensorflow.org/models/$(MOBILENET_CKPT_TAR);)
	(cd model; tar -xvf $(MOBILENET_CKPT_TAR);)

check_model:
ifndef TF_MODELS_PATH
	$(error TF_MODELS_PATH is not defined. Run `export TF_MODELS_PATH=path/to/your/tensorflow/models/repo`)
endif

.PHONY: export
export: check_model
	@echo "\nExporting model for inference..."
	(cd model; python3 $(TF_MODELS_PATH)/research/slim/export_inference_graph.py \
             --alsologtostderr \
             --model_name=$(MOBILENET_NAME) \
             --batch_size=1 \
             --image_size=$(MOBILENET_SIZE) \
             --output_file=$(MOBILENET_CKPT).pb;)

check_tf:
ifndef TF_SRC_PATH
	$(error TF_SRC_PATH is not defined. Run `export TF_SRC_PATH=path/to/your/tensorflow/source/repo`)
endif

.PHONY: freeze
freeze: check_tf
	@echo "\nFreezing model for inference..." 
	(cd model; python3 $(TF_SRC_PATH)/tensorflow/python/tools/freeze_graph.py \
             --input_graph=$(MOBILENET_CKPT).pb \
             --input_binary=true \
             --input_checkpoint=$(MOBILENET_CKPT).ckpt \
             --output_graph=$(MOBILENET_CKPT)_frozen.pb \
             --output_node_name=MobilenetV1/Predictions/Reshape_1;)

.PHONY: compile
compile:
	@echo "\nCompiling model to Movidius graph..."
	(cd model; mvNCCompile -s 12 $(MOBILENET_CKPT)_frozen.pb -in=input -on=MobilenetV1/Predictions/Reshape_1;)

.PHONY: profile
profile:
	@echo "\nProfiling the model..."
	(cd model; mvNCProfile -s 12 $(MOBILENET_CKPT)_frozen.pb -in=input -on=MobilenetV1/Predictions/Reshape_1;)

.PHONY: check
check:
	@echo "\nComparing results with standard Tensorflow..."
	(cd model; mvNCCheck -s 12 $(MOBILENET_CKPT)_frozen.pb -in=input -on=MobilenetV1/Predictions/Reshape_1;)

.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo "  make help - Shows this message.";
	@echo "  make - Builds all dependencies, but does not run this program.";
	@echo "  make checkpoint - Downloads pre-trained checkpoint files.;"
	@echo "  make export - Export the neural network model for inference.";
	@echo "  make freeze - Freeze the neural network model for inference.";
	@echo "  make compile - Convert the frozen model into Movidius graph file.";
	@echo "  make profile - Run the model on NCS and extract complexity, bandwidth and execution time for each layer.";

.PHONY: clean
clean:
	@echo "\nMaking clean...";
	rm -rf model
