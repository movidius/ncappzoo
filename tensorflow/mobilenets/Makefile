NCCOMPILE = mo.py

YELLOW='\033[1;33m'
NOCOLOR='\033[0m'
RED = '\033[1;31m'

VERSION?=v1
DEPTH?=1.0
IMGSIZE?=224
MODEL_CKPT=mobilenet_$(VERSION)_$(DEPTH)_$(IMGSIZE)
MODEL_CKPT_TAR=$(MODEL_CKPT)_2017_06_14.tar.gz
LABELS_FILE = categories.txt

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

SIMPLE_CLASSIFIER_CPP_RELATIVE_DIR = ../../apps/simple_classifier_cpp
SIMPLE_CLASSIFIER_PY_RELATIVE_DIR = ../../apps/simple_classifier_py
MODEL_RELATIVE_DIR = ./model
TEST_IMAGE_RELATIVE_DIR = ../../data/images
GRAPH_FILENAME_BASE = $(MODEL_CKPT)_frozen
TEST_IMAGE_FILENAME = nps_electric_guitar.png

.PHONY: all
all: deps data compile_model compile_cpp

PROJECT_DIR = $(CURDIR)
ZOO_RELATIVE_TF_SRC_PROJ = ../tf_src
ZOO_RELATIVE_TF_MODELS_PROJ = ../tf_models
ZOO_TF_SRC_PATH = ${PROJECT_DIR}/${ZOO_RELATIVE_TF_SRC_PROJ}/tensorflow
ZOO_TF_MODELS_PATH = ${PROJECT_DIR}/${ZOO_RELATIVE_TF_MODELS_PROJ}/models

MEAN_VALUES = [128,128,128]
SCALE = 128

.PHONY: deps
deps: freeze
	@echo $(YELLOW)"\nMobilenet: Making dependencies..."$(NOCOLOR)
	

.PHONY: data
data: 
	@echo $(YELLOW)"\nMobilenet: Downloading required data..."$(NOCOLOR)
	(cd ../../data/ilsvrc12; make;); \
	(mkdir -p model); \
	(cd model; wget -nc http://download.tensorflow.org/models/$(MODEL_CKPT_TAR)); \
	(cd model; tar -xvf $(MODEL_CKPT_TAR);); \


# Checks if TF_MODELS_PATH is set.  If it isn't set then will make the
# tf_models project in the ncappzoo/tensorflow/tf_models directory.
check_model:
ifndef TF_MODELS_PATH
	@echo $(YELLOW)"\nTF_MODELS_PATH not set, making tf_models..."$(NOCOLOR)
	(cd ../tf_models; make all; cd ${PROJECT_DIR})
	$(eval TF_MODELS_PATH = ${ZOO_TF_MODELS_PATH})
	@echo $(YELLOW)"TF_MODELS_PATH is ${TF_MODELS_PATH}"$(NOCOLOR)
else
	@echo $(YELLOW)"TF_MODELS_PATH is ${TF_MODELS_PATH}"$(NOCOLOR)
endif

# Checks if TF_SRC_PATH is set.  If it isn't set then will make the
# tf_models project in the ncappzoo/tensorflow/tf_src directory.
check_tf:
ifndef TF_SRC_PATH
	@echo $(YELLOW)"\nTF_SRC_PATH not set, making tf_src..."$(NOCOLOR)
	(cd ../tf_src; make all; cd ${PROJECT_DIR})
	$(eval TF_SRC_PATH=${ZOO_TF_SRC_PATH})
	@echo $(YELLOW)"TF_SRC_PATH is ${TF_SRC_PATH}"$(NOCOLOR) 
else
	@echo $(YELLOW)"TF_SRC_PATH is ${TF_SRC_PATH}"$(NOCOLOR) 
endif


.PHONY: export
export: check_model
	@echo $(YELLOW)"\nMobilenet: Exporting GraphDef file..."$(NOCOLOR)
	@if [ -e ${MODEL_RELATIVE_DIR}/${MODEL_CKPT}.pb ] ; \
	then \
		echo " - No export needed."; \
	else \
		echo " - Frozen model not found. Exporting model to pb..."; \
		(cd model; python3 ${TF_MODELS_PATH}/research/slim/export_inference_graph.py \
			--alsologtostderr \
			--model_name=$(MODEL_NAME) \
			--batch_size=1 \
			--dataset_name=imagenet \
			--image_size=$(IMGSIZE) \
			--output_file=$(MODEL_CKPT).pb;) ; \
	fi
	
	
.PHONY: freeze
freeze: check_tf data export
	@echo $(YELLOW)"\nMobilenet: Freezing model for inference..."$(NOCOLOR)
	@if [ -e ${MODEL_RELATIVE_DIR}/${MODEL_CKPT}_frozen.pb ] ; \
	then \
		echo " - Frozen model file already exists, skipping freeze."; \
	else \
		(cd model; python3 $(TF_SRC_PATH)/tensorflow/python/tools/freeze_graph.py \
			--input_graph=$(MODEL_CKPT).pb \
			--input_binary=true \
			--input_checkpoint=$(MODEL_CKPT).ckpt \
			--output_graph=$(MODEL_CKPT)_frozen.pb \
			--output_node_name=$(OUTPUT_NODE_NAME);); \
	fi


.PHONY: compile_model
compile_model: freeze
	@echo $(YELLOW)"\nMobilenet: Compiling model to IR..."$(NOCOLOR)
	@if [ -e ${MODEL_RELATIVE_DIR}/${GRAPH_FILENAME_BASE}.xml ] && [ -e ${MODEL_RELATIVE_DIR}/${GRAPH_FILENAME_BASE}.bin ]; \
	then \
		echo " - Compiled model file already exists, skipping compile."; \
	else \
		echo " - IR does not exist. Compiling model files to IR..."; \
		cd ${MODEL_RELATIVE_DIR}; \
		${NCCOMPILE} --data_type=FP16 --reverse_input_channels --scale=${SCALE} --mean_values=${MEAN_VALUES} --input_model=$(MODEL_CKPT)_frozen.pb || (echo $(RED)"Make sure to set the OpenVINO environment variables using the "$(YELLOW)"setupvars.sh"$(RED)" script found in <your OpenVINO install location>/bin/ folder."$(NOCOLOR);  exit 1;); \
	fi


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\nMobilenet: Checking application requirements...\n"$(NOCOLOR)
	@echo "No requirements needed."
	
	
.PHONY: run
run: run_py 


.PHONY:	compile_cpp
compile_cpp:  
	@echo $(YELLOW)"\nMobilenet: Compiling C++ sample..."$(NOCOLOR)
	(cd ${SIMPLE_CLASSIFIER_CPP_RELATIVE_DIR}; make compile_cpp; cd -)


.PHONY: run_cpp
run_cpp: install-reqs data deps compile_cpp compile_model
	@echo $(YELLOW)"\nMobilenet: Running C++ sample..."$(NOCOLOR)
	${SIMPLE_CLASSIFIER_CPP_RELATIVE_DIR}/build/simple_classifier_cpp ${MODEL_RELATIVE_DIR}/${GRAPH_FILENAME_BASE}.xml ${TEST_IMAGE_RELATIVE_DIR}/${TEST_IMAGE_FILENAME} ${LABELS_FILE}


.PHONY: run_py
run_py: install-reqs data deps compile_model
	@echo $(YELLOW)"\nMobilenet: Running Python sample..."$(NOCOLOR)
	python3 ${SIMPLE_CLASSIFIER_PY_RELATIVE_DIR}/run.py --ir=${MODEL_RELATIVE_DIR}/${GRAPH_FILENAME_BASE}.xml --image=${TEST_IMAGE_RELATIVE_DIR}/${TEST_IMAGE_FILENAME} --labels=${LABELS_FILE}
	

.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run "$(NOCOLOR)"- Runs Python example.";
	@echo $(YELLOW)"  make run_py "$(NOCOLOR)"- Runs Python example program.";
	@echo $(YELLOW)"  make run_cpp "$(NOCOLOR)"- Builds and runs C++ example program.";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- Shows this message.";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- Makes everything needed to run, but doesn't run.";
	@echo $(YELLOW)"  make compile_cpp "$(NOCOLOR)"- Builds C++ example to demonstrate the network.";
	@echo $(YELLOW)"  make compile_model "$(NOCOLOR)"- Uses the Model Optimizer to compile the model files to IR.";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- Downloads required data.";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- Makes dependencies for project, prepares model etc.";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- Removes all created content."
	@echo "  ";
	@echo $(YELLOW)"  Note"$(NOCOLOR)": by default will use Mobilenet v1: depth multiplier: 1.0, image size: 224";
	@echo " ";
	@echo $(NOCOLOR)"  Full command list to run other Mobilenet v1 variants:";
	@echo $(YELLOW)"    make run DEPTH=1.0 IMGSIZE=224" $(NOCOLOR);
	@echo $(YELLOW)"    make run DEPTH=1.0 IMGSIZE=192" $(NOCOLOR);
	@echo $(YELLOW)"    make run DEPTH=1.0 IMGSIZE=160" $(NOCOLOR);
	@echo $(YELLOW)"    make run DEPTH=1.0 IMGSIZE=128" $(NOCOLOR);
	@echo $(YELLOW)"    make run DEPTH=0.75 IMGSIZE=224" $(NOCOLOR);
	@echo $(YELLOW)"    make run DEPTH=0.75 IMGSIZE=192" $(NOCOLOR);
	@echo $(YELLOW)"    make run DEPTH=0.75 IMGSIZE=160" $(NOCOLOR);
	@echo $(YELLOW)"    make run DEPTH=0.75 IMGSIZE=128" $(NOCOLOR);
	@echo $(YELLOW)"    make run DEPTH=0.50 IMGSIZE=224" $(NOCOLOR);
	@echo $(YELLOW)"    make run DEPTH=0.50 IMGSIZE=192" $(NOCOLOR);
	@echo $(YELLOW)"    make run DEPTH=0.50 IMGSIZE=160" $(NOCOLOR);
	@echo $(YELLOW)"    make run DEPTH=0.50 IMGSIZE=128" $(NOCOLOR);
	@echo $(YELLOW)"    make run DEPTH=0.25 IMGSIZE=224 "$(NOCOLOR);
	@echo $(YELLOW)"    make run DEPTH=0.25 IMGSIZE=192" $(NOCOLOR);
	@echo $(YELLOW)"    make run DEPTH=0.25 IMGSIZE=160" $(NOCOLOR);
	@echo $(YELLOW)"    make run DEPTH=0.25 IMGSIZE=128" $(NOCOLOR);
	@echo " "


.PHONY: clean
clean:
	@echo $(YELLOW)"\nMobilenet: Cleaning up files..."$(NOCOLOR);
	rm -rf model
	rm -f *.pb
