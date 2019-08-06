NCCOMPILE = mo.py

GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'

CLASSIFIER_RELATIVE_DIR = ../../apps/simple_classifier_py
TEST_IMAGE_RELATIVE_DIR = ../../data/images
LABELS_RELATIVE_DIR = ../../data/ilsvrc12
MODEL_RELATIVE_DIR = ../../caffe/GoogLeNet

TEST_IMAGE_FILENAME = nps_electric_guitar.png
LABELS_FILENAME = synset_labels.txt

MODEL_RELATIVE_DIR = ../../caffe/GoogLeNet
MODEL_FILE_NAME_BASE = googlenet-v1


.PHONY: all
all: deps data


.PHONY: deps
deps: googlenet_compile_model 
	@echo $(YELLOW)"\nsimple_classifier_py: Making dependencies..."$(NOCOLOR)


.PHONY: data
data:
	@echo $(YELLOW)"\nsimple_classifier_py: Making the required data..."$(NOCOLOR)
	@(cd ../../data/ilsvrc12; make all;)


.PHONY: googlenet_compile_model
googlenet_compile_model: 
	@echo $(YELLOW)"\nsimple_classifier_py: Compiling the model to IR..."$(NOCOLOR)
	@if [ -e ${MODEL_FILE_NAME_BASE}.bin ] && [ -e ${MODEL_FILE_NAME_BASE}.xml ]; \
	then \
		echo "- googlenet IR files found. No need to compile."; \
	else \
		echo "- making googlenet IRs in googlenet project."; \
		echo "- copying IRs to project folder..."; \
		(cd ${MODEL_RELATIVE_DIR}; make compile_model; cp ${MODEL_FILE_NAME_BASE}.bin ${CLASSIFIER_RELATIVE_DIR}; cp ${MODEL_FILE_NAME_BASE}.xml ${CLASSIFIER_RELATIVE_DIR};); \
	fi 


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\nsimple_classifier_py: Checking application requirements...\n"$(NOCOLOR)
	@echo "No requirements needed."
	
	
.PHONY: run
run: run_py


.PHONY: run_py
run_py: install-reqs deps data
	@echo $(YELLOW)"\nsimple_classifier_py: Running the sample..."$(NOCOLOR)
	python3 run.py --ir=${MODEL_FILE_NAME_BASE}.xml --image=${TEST_IMAGE_RELATIVE_DIR}/${TEST_IMAGE_FILENAME} --labels=${LABELS_RELATIVE_DIR}/${LABELS_FILENAME} 


.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run "$(NOCOLOR)"- Runs the provided run.py file which sends a single image to the Neural Compute Stick and receives and displays the inference results.";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- Shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- Makes everything needed to run, but doesn't run";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- Downloads the required data.";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- Makes dependencies for project, downloads and prepares model etc.";
	@echo $(YELLOW)"  make prototxt "$(NOCOLOR)"- Downloads and adds input shape to Caffe prototxt file";
	@echo $(YELLOW)"  make caffemodel "$(NOCOLOR)"- Downloads the caffemodel for the network";
	@echo $(YELLOW)"  make compile_model "$(NOCOLOR)"- Compiles the model to an IR format.";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- Removes all the temporary files that are created by the Makefile.\n";


clean: 
	@echo $(YELLOW)"\nsimple_classifier_py: Cleaning up files..."$(NOCOLOR)
	rm -f ${MODEL_FILE_NAME_BASE}.xml
	rm -f ${MODEL_FILE_NAME_BASE}.bin
