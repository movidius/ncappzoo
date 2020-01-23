NCCOMPILE = mo.py

GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'

SIMPLE_CLASSIFIER_CPP_RELATIVE_DIR = ../../apps/simple_classifier_cpp
SIMPLE_CLASSIFIER_PY_RELATIVE_DIR = ../../apps/simple_classifier_py

IMAGE_RELATIVE_DIR = ../../data/images
LABELS_RELATIVE_DIR = ../../data/ilsvrc12
MODEL_RELATIVE_DIR = ../../caffe/AlexNet

IMAGE_FILENAME = nps_electric_guitar.png
LABELS_FILENAME = synset_labels.txt

ZOO_RELATIVE_DIR = ../../omz
MODEL_DOWNLOADER_DIR = open_model_zoo/tools/downloader
MODEL_DOWNLOADER_FILENAME = downloader.py
ALEXNET_MODEL_ZOO_DIR = classification/alexnet/caffe

MODEL_FILE_NAME_BASE = alexnet
MEAN_VALUES = [104,117,123]

IMAGE = ${IMAGE_RELATIVE_DIR}/${IMAGE_FILENAME}
LABELS = ${LABELS_RELATIVE_DIR}/${LABELS_FILENAME}
XML = ${MODEL_FILE_NAME_BASE}.xml

NETWORK_NAME = alexnet

.PHONY: all
all: deps data compile_model compile_cpp


.PHONY: deps
deps: download_model_files
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Making dependencies..."$(NOCOLOR)

	
.PHONY: data
data:
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Downloading required data..."$(NOCOLOR)
	(cd ../../data/ilsvrc12; make all;)


.PHONY: model_zoo
model_zoo:
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Making model zoo..."$(NOCOLOR)
	(cd ${ZOO_RELATIVE_DIR}; make all;)


.PHONY: download_model_files
download_model_files: model_zoo
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Checking zoo model files..."$(NOCOLOR)
	@if [ -e ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${ALEXNET_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.caffemodel ] || [ -e ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${ALEXNET_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.prototxt ] ;\
	then \
		echo " - Model files already exists." ; \
	else \
		echo " - Model files do not exist. Using Model downloader to download the model..." ; \
		(cd ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}; python3 ${MODEL_DOWNLOADER_FILENAME} --name ${MODEL_FILE_NAME_BASE};); \
	fi


.PHONY: compile_model
compile_model: download_model_files
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Compiling model to IR..."$(NOCOLOR)
	@echo "Checking OpenVINO environment..."
	@if [ -z "$(INTEL_OPENVINO_DIR)" ] ; \
	then \
		echo "Please initiate the Intel OpenVINO environment by going to the installation directory for openvino and running the setupvars.sh file in the bin folder." ; \
		exit 1 ; \
	else \
		echo "Intel OpenVINO environment is already set!" ; \
	fi
	@if [ -e ${MODEL_FILE_NAME_BASE}.xml ] && [ -e ${MODEL_FILE_NAME_BASE}.bin ]; \
	then \
		echo " - IR files already exist. Skipping compile step."; \
	else \
		${NCCOMPILE} --data_type=FP16 --input_model=${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${ALEXNET_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.caffemodel --input_proto=${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${ALEXNET_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.prototxt --mean_values=${MEAN_VALUES}; \
	fi 


.PHONY: run
run: run_py


.PHONY:	compile_cpp
compile_cpp:  
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Compiling C++ sample..."$(NOCOLOR)
	@echo "Checking OpenVINO environment..."
	@if [ -z "$(INTEL_OPENVINO_DIR)" ] ; \
	then \
		echo "Please initiate the Intel OpenVINO environment by going to the installation directory for openvino and running the setupvars.sh file in the bin folder." ; \
		exit 1 ; \
	else \
		echo "Intel OpenVINO environment is already set!" ; \
	fi
	(cd ${SIMPLE_CLASSIFIER_CPP_RELATIVE_DIR}; make compile_cpp; cd -)


.PHONY: run_cpp
run_cpp: deps data compile_cpp compile_model
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Running C++ sample..."$(NOCOLOR)
	${SIMPLE_CLASSIFIER_CPP_RELATIVE_DIR}/build/simple_classifier_cpp ${XML} ${IMAGE} ${LABELS};

	
.PHONY: run_py
run_py: deps data compile_model
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Running Python sample..."$(NOCOLOR)
	python3 ${SIMPLE_CLASSIFIER_PY_RELATIVE_DIR}/run.py --ir=${XML} --image=${IMAGE} --labels=${LABELS} --top=5; 


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Checking application requirements...\n"$(NOCOLOR)
	@echo "No requirements needed."


.PHONY: uninstall-reqs
uninstall-reqs: 
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Uninstalling requirements..."$(NOCOLOR)
	@echo "Nothing to uninstall."
	
	
.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run or run_py "$(NOCOLOR)"- Runs the Python example.";
	@echo $(YELLOW)"  make run_cpp "$(NOCOLOR)"- Runs the C++ example.";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- Shows this message.";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- Makes everything needed to run, but doesn't run.";
	@echo $(YELLOW)"  make compile_model "$(NOCOLOR)"- Compiles the model to an IR format.";
	@echo $(YELLOW)"  make compile_cpp "$(NOCOLOR)"- Builds the C++ example.";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- Downloads the required data.";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- Makes dependencies for project, downloads and prepares model etc.";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- Removes all temporary files that are created by the Makefile.\n";
	@echo " ----- "$(YELLOW)"Model information"$(NOCOLOR)" -----"
	@echo $(YELLOW)" Description: "$(NOCOLOR)
	@echo "  This network does image classification on 1000 different classes.";
	@echo $(YELLOW)" Model inputs:"$(NOCOLOR)
	@echo "  name: 'data', shape: [1x3x227x227], Expected color order is BGR.";
	@echo $(YELLOW)" Model outputs:"$(NOCOLOR)
	@echo "  name: 'prob', shape: [1, 1000] - Output indexes represent each class probability.\n";
	

clean: 
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Cleaning up files..."$(NOCOLOR)
	rm -f ${MODEL_FILE_NAME_BASE}.bin
	rm -f ${MODEL_FILE_NAME_BASE}.xml
	rm -f ${MODEL_FILE_NAME_BASE}.mapping
