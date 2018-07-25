
#NCCOMPILE = python /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo.py
NCCOMPILE = mo.py

MODEL_NAME=inception_resnet_v2
MODEL_CKPT=$(MODEL_NAME)
IMGSIZE=299
OUTPUT_NODE_NAME=InceptionResnetV2/Logits/Predictions
MODEL_CKPT_TAR=$(MODEL_CKPT)_2016_08_30.tar.gz
MODEL_CKPT_AFTER_TAR=$(MODEL_CKPT)_2016_08_30.ckpt

GRAPH_FILENAME_BASE = inception_resnet_v2_frozen

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
	@if [ -e $(MODEL_CKPT_AFTER_TAR) ] ; \
	then \
		echo "checkpoint file already exists, skipping checkpoint."; \
	else \
		wget -nc http://download.tensorflow.org/models/$(MODEL_CKPT_TAR); \
		tar -xvf $(MODEL_CKPT_TAR); \
		rm -f $(MODEL_CKPT_TAR); \
	fi

.PHONY: export
export: check_model
	@echo $(YELLOW)"\nExporting GraphDef file..."$(NOCOLOR)
	@if [ -e $(MODEL_CKPT).pb ] ; \
	then \
		echo "no export needed"; \
	else \
		python3 $(TF_MODELS_PATH)/research/slim/export_inference_graph.py \
				--alsologtostderr \
				--model_name=$(MODEL_NAME) \
				--batch_size=1 \
				--dataset_name=imagenet \
				--image_size=$(IMGSIZE) \
				--output_file=$(MODEL_CKPT).pb; \
	fi

.PHONY: freeze
freeze: check_tf checkpoint export
	@echo $(YELLOW)"\nFreezing model for inference..."$(NOCOLOR)
	@if [ -e $(MODEL_CKPT)_frozen.pb ] ; \
	then \
		echo "frozen model file already exists, skipping freeze."; \
	else \
		python3 $(TF_SRC_PATH)/tensorflow/python/tools/freeze_graph.py \
			--input_graph=$(MODEL_CKPT).pb \
			--input_binary=true \
			--input_checkpoint=$(MODEL_CKPT)_2016_08_30.ckpt \
			--output_graph=$(MODEL_CKPT)_frozen.pb \
			--output_node_name=$(OUTPUT_NODE_NAME); \
	fi

.PHONY: compile_model
compile_model: deps
	@echo "\nmaking compile"; \
	if [ -e ${GRAPH_FILENAME_BASE}.xml ] ; \
	then \
		echo "compiled model file already exists, skipping compile."; \
	else \
		${NCCOMPILE} --data_type FP16 --input_model $(MODEL_CKPT)_frozen.pb ;\
	fi



.PHONY: check
check: 
	@echo "\nCheck not implemented yet."

.PHONY: data
data: 
	@echo "\nmaking data - no data required for this project"

.PHONY: profile
profile: 
	@echo "\nProfile not implemented yet"

.PHONY: run
run: run_cpp 
	@echo "\nmaking run"

.PHONY:	compile_cpp
compile_cpp:  
	@echo "\nmaking compile_cpp"
	(cd ${SIMPLE_CLASSIFIER_RELATIVE_DIR}; make compile_cpp; cd -)

.PHONY: run_cpp
run_cpp: all
	@echo "\nmaking run_cpp"
	${SIMPLE_CLASSIFIER_RELATIVE_DIR}/build/simple_classifier ${GRAPH_FILENAME_BASE}.xml ${IMAGES_RELATIVE_DIR}/${TEST_IMAGE_FILENAME}; 



.PHONY: help
help:
	@echo $(YELLOW)"\nPossible make targets: "$(NOCOLOR);
	@echo "  make help - Shows this message.";
	@echo "  make all - make all - everything needed to run, but doesn't run";
	@echo "  make clean - removes files created in this directory.";
	@echo "  make compile_cpp - builds C++ example to demonstrate the network ";
	@echo "  make compile_model - runs model compiler for the network";
	@echo "  make data - downloads required data.";
	@echo "  make deps - makes dependencies for project, prepares model etc.";
	@echo "  make run - builds and runs C++ example program";
	@echo "  make clean - removes all created content"

.PHONY: clean
clean:
	@echo $(YELLOW)"\nMaking clean..."$(NOCOLOR);
	rm -f inception_resnet_v2* -v
	rm -f ${GRAPH_FILENAME_BASE}.bin
	rm -f ${GRAPH_FILENAME_BASE}.mapping.json
	rm -f ${GRAPH_FILENAME_BASE}.xml
	rm -f labels.txt
