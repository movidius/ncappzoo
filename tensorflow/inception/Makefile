#NCCOMPILE = python /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo.py
NCCOMPILE = mo.py

VERSION?=v3
MODEL_CKPT=$(MODEL_NAME)

# Inception model names as defined in https://github.com/tensorflow/models/blob/master/research/slim/nets/nets_factory.py
MODEL_NAME=inception_$(VERSION)

ifeq ($(VERSION),v1)
IMGSIZE=224
MODEL_CKPT_TAR=$(MODEL_CKPT)_2016_08_28.tar.gz
MODEL_CKPT_AFTER_TAR=$(MODEL_CKPT)_2016_08_28.ckpt
OUTPUT_NODE_NAME=InceptionV1/Logits/Predictions/Reshape_1
else ifeq ($(VERSION),v2)
IMGSIZE=224
MODEL_CKPT_TAR=$(MODEL_CKPT)_2016_08_28.tar.gz
MODEL_CKPT_AFTER_TAR=$(MODEL_CKPT)_2016_08_28.ckpt
OUTPUT_NODE_NAME=InceptionV2/Predictions/Reshape_1
else ifeq ($(VERSION),v3)
IMGSIZE=299
MODEL_CKPT_TAR=$(MODEL_CKPT)_2016_08_28.tar.gz
MODEL_CKPT_AFTER_TAR=$(MODEL_CKPT)_2016_08_28.ckpt
OUTPUT_NODE_NAME=InceptionV3/Predictions/Reshape_1
else ifeq ($(VERSION),v4)
IMGSIZE=299
MODEL_CKPT_TAR=$(MODEL_CKPT)_2016_09_09.tar.gz
MODEL_CKPT_AFTER_TAR=$(MODEL_CKPT)_2016_09_09.ckpt
OUTPUT_NODE_NAME=InceptionV4/Logits/Predictions
endif

MODEL_RELATIVE_DIR=model/$(VERSION)

GRAPH_FILENAME_BASE = inception_$(VERSION)_frozen

SIMPLE_CLASSIFIER_RELATIVE_DIR = ../../apps/simple_classifier
IMAGES_RELATIVE_DIR = ../../data/images
TEST_IMAGE_FILENAME = nps_electric_guitar.png



PROJECT_DIR = $(CURDIR)
ZOO_RELATIVE_TF_SRC_PROJ = ../tf_src
ZOO_RELATIVE_TF_MODELS_PROJ = ../tf_models
ZOO_TF_SRC_PATH = ${PROJECT_DIR}/${ZOO_RELATIVE_TF_SRC_PROJ}/tensorflow
ZOO_TF_MODELS_PATH = ${PROJECT_DIR}/${ZOO_RELATIVE_TF_MODELS_PROJ}/models

YELLOW='\033[1;33m'
NOCOLOR='\033[0m'

.PHONY: all
all: deps data compile_model compile_cpp

# Checks if TF_MODELS_PATH is set.  If it isn't set then will make the
# tf_models project in the ncappzoo/tensorflow/tf_models directory.
check_model:
ifndef TF_MODELS_PATH
	@echo "TF_MODELS_PATH not set, making tf_models";
	(cd ../tf_models; make all; cd ${PROJECT_DIR})
	$(eval TF_MODELS_PATH = ${ZOO_TF_MODELS_PATH})
	@echo "TF_MODELS_PATH is ${TF_MODELS_PATH}" ;
else
	@echo "TF_MODELS_PATH is ${TF_MODELS_PATH}" ;
endif

# Checks if TF_SRC_PATH is set.  If it isn't set then will make the
# tf_models project in the ncappzoo/tensorflow/tf_src directory.
check_tf:
ifndef TF_SRC_PATH
	@echo "TF_SRC_PATH not set, making tf_src";
	(cd ../tf_src; make all; cd ${PROJECT_DIR})
	$(eval TF_SRC_PATH=${ZOO_TF_SRC_PATH})
	@echo "TF_SRC_PATH is ${TF_SRC_PATH}" ;
else
	@echo "TF_SRC_PATH is ${TF_SRC_PATH}" ;
endif

.PHONY: deps
deps: freeze
	@echo "\nmaking deps"

.PHONY: checkpoint
checkpoint:
	@echo $(YELLOW)"\nDownloading checkpoint files..."$(NOCOLOR)
	@if [ -e $(MODEL_RELATIVE_DIR)/$(MODEL_CKPT_AFTER_TAR) ] ; \
	then \
		echo "checkpoint file already exists, skipping checkpoint."; \
	else \
		mkdir -p model/$(VERSION); \
		cd $(MODEL_RELATIVE_DIR); \
		wget -nc http://download.tensorflow.org/models/$(MODEL_CKPT_TAR); \
		tar -xvf $(MODEL_CKPT_TAR); \
		cd ../..; \
	fi;

.PHONY: export
export: check_model
	@echo $(YELLOW)"\nExporting GraphDef file..."$(NOCOLOR)
	@if [ -e $(MODEL_RELATIVE_DIR)/$(MODEL_CKPT).pb ] ; \
	then \
		echo "no export needed"; \
	else \
		cd $(MODEL_RELATIVE_DIR); \
		python3 $(TF_MODELS_PATH)/research/slim/export_inference_graph.py \
			--alsologtostderr \
			--model_name=$(MODEL_NAME) \
			--batch_size=1 \
			--dataset_name=imagenet \
			--image_size=$(IMGSIZE) \
			--output_file=$(MODEL_CKPT).pb; \
		cd ../..; \
	fi;

.PHONY: freeze
freeze: check_tf checkpoint export
	@echo $(YELLOW)"\nFreezing model for inference..."$(NOCOLOR)
	@if [ -e $(MODEL_RELATIVE_DIR)/$(MODEL_CKPT)_frozen.pb ] ; \
	then \
		echo "frozen model file already exists, skipping freeze."; \
	else \
		cd $(MODEL_RELATIVE_DIR); \
		python3 $(TF_SRC_PATH)/tensorflow/python/tools/freeze_graph.py \
			--input_graph=$(MODEL_CKPT).pb \
			--input_binary=true \
			--input_checkpoint=$(MODEL_CKPT).ckpt \
			--output_graph=$(MODEL_CKPT)_frozen.pb \
			--output_node_name=$(OUTPUT_NODE_NAME); \
		cd ../..; \
	fi;

.PHONY: compile_model
compile_model: deps
	@echo $(YELLOW)"\nCompiling model to Movidius graph..."$(NOCOLOR)
	@if [ -e ${MODEL_RELATIVE_DIR}/${GRAPH_FILENAME_BASE}.xml ] ; \
	then \
		echo "compiled model file already exists, skipping compile."; \
	else \
		cd ${MODEL_RELATIVE_DIR}; \
		${NCCOMPILE} --data_type FP16 --input_model $(MODEL_CKPT)_frozen.pb ; \
		echo $(YELLOW)"Graph generated!\n"$(NOCOLOR) ;\
	fi;

# old  compile line
#mvNCCompile -s 12 $(MODEL_CKPT)_frozen.pb -in=input -on=$(OUTPUT_NODE_NAME);


.PHONY: profile
profile: freeze
	@echo $(YELLOW)"\nProfiling the model..."$(NOCOLOR)
	(cd model/$(VERSION); mvNCProfile -s 12 $(MODEL_CKPT)_frozen.pb -in=input -on=$(OUTPUT_NODE_NAME);)
	@echo $(YELLOW)"Movidius graph generated! You can run inferences using ncappzoo/apps/image-classifier project.\n"$(NOCOLOR)

.PHONY: check
check: freeze
	@echo $(YELLOW)"\nComparing results with standard Tensorflow..."$(NOCOLOR)
	(cd model/$(VERSION); mvNCCheck -s 12 $(MODEL_CKPT)_frozen.pb -in=input -on=$(OUTPUT_NODE_NAME);)
	@echo $(YELLOW)"Movidius graph generated! You can run inferences using ncappzoo/apps/image-classifier project.\n"$(NOCOLOR)

.PHONY: run
run: compile
	@echo $(YELLOW)"\nRunning inferences using image-classifier project..."$(NOCOLOR)
	(python3 ../../apps/image-classifier/image-classifier.py --graph ./model/$(VERSION)/graph --dim $(IMGSIZE) $(IMGSIZE) --mean 127.5 --scale 0.00789 --colormode "RGB" --labels ./model/$(VERSION)/labels.txt --image ../../data/images/pic_053.jpg)

.PHONY: help
help:
	@echo $(YELLOW)"\nPossible make targets: "$(NOCOLOR);
	@echo "  make help - Shows this message.";
	@echo "  make all - make all - everything needed to run, but doesn't run.";
	@echo "  make clean - removes files created in this directory.";
	@echo "  make compile_cpp - builds C++ example to demonstrate the network.";
	@echo "  make compile_model - runs model compiler for the network.";
	@echo "  make data - downloads required data.";
	@echo "  make deps - makes dependencies for project, prepares model etc.";
	@echo "  make run - builds and runs C++ example program.";
	@echo "  make clean - removes all created content."
	@echo "  ";
	@echo "  Note: by default inception V3 will be used for all of the above commands, but other versions";
	@echo "        of inception can be specified by setting VERSION on the command line like this:";
	@echo "        to make all for Inception V1 use make VERSION=v1 (or make all VERSION=v1)";
	@echo "        to compile the model for Inception V1 use make compile_model VERSION=v1"

.PHONY: clean
clean:
	@echo $(YELLOW)"\nMaking clean..."$(NOCOLOR);
	rm -rf model
	rm -rf labels.txt
	rm -rf inception_v3.pb
