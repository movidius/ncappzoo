
GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'

CLASSIFIER_RELATIVE_DIR = ../../apps/simple_classifier_cpp
GOOGLENET_RELATIVE_DIR = ../../caffe/GoogLeNet
GOOGLENET_BASE_NAME = googlenet-v1

IMAGES_RELATIVE_DIR = ../../data/images
TEST_IMAGE_FILENAME = nps_electric_guitar.png

LABELS_RELATIVE_DIR = ../../data/ilsvrc12
LABELS_FILENAME = synset_labels.txt

.PHONY: all
all: deps data compile_cpp


.PHONY: googlenet_data
googlenet_data: 
	@echo $(YELLOW)"\nsimple_classifier_cpp: making googlenet_data..."$(NOCOLOR)
	(cd ${GOOGLENET_RELATIVE_DIR}; make data; cd -) 


.PHONY: googlenet_deps
googlenet_deps: 
	@echo $(YELLOW)"\nsimple_classifier_cpp: making googlenet_deps..."$(NOCOLOR)
	(cd ${GOOGLENET_RELATIVE_DIR}; make deps; cd -) 


.PHONY: googlenet_compile_model
googlenet_compile_model: 
	@echo $(YELLOW)"\nsimple_classifier_cpp: Compiling the model to IR..."$(NOCOLOR)
	@if [ -e ${GOOGLENET_BASE_NAME}.bin ] && [ -e ${GOOGLENET_BASE_NAME}.xml ]; \
	then \
		echo "- googlenet IR files found. No need to compile."; \
	else \
		echo "- making googlenet IRs in googlenet project."; \
		echo "- copying IRs to project folder..."; \
		(cd ${GOOGLENET_RELATIVE_DIR}; make compile_model; cp ${GOOGLENET_BASE_NAME}.bin ${CLASSIFIER_RELATIVE_DIR}; cp ${GOOGLENET_BASE_NAME}.xml ${CLASSIFIER_RELATIVE_DIR};); \
	fi 


.PHONY: data
data: googlenet_data
	@echo $(YELLOW)"\nsimple_classifier_cpp: Downloading the required data..."$(NOCOLOR)
	(cd ${LABELS_RELATIVE_DIR}; make;)


.PHONY: deps
deps: googlenet_deps googlenet_compile_model
	@echo $(YELLOW)"\nsimple_classifier_cpp: Making dependencies..."$(NOCOLOR)


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\nsimple_classifier_cpp: Checking application requirements...\n"$(NOCOLOR)
	@echo "No requirements needed."
	

.PHONY: run
run: run_cpp
	@echo $(YELLOW)"\nsimple_classifier_cpp: Running the sample..."$(NOCOLOR)


.PHONY:	compile_cpp
compile_cpp:  
	@echo $(YELLOW)"\nsimple_classifier_cpp: Building the sample..."$(NOCOLOR)
	(mkdir -p build; cd build; cmake ..; make)


.PHONY: run_cpp
run_cpp: install-reqs deps data compile_cpp
	@echo $(YELLOW)"\nsimple_classifier_cpp: Running the sample..."$(NOCOLOR)
	./build/simple_classifier_cpp ${GOOGLENET_BASE_NAME}.xml ${IMAGES_RELATIVE_DIR}/${TEST_IMAGE_FILENAME} ${LABELS_RELATIVE_DIR}/${LABELS_FILENAME}


.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run_cpp "$(NOCOLOR)"- builds and runs the C++ example program";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- everything needed to run, but doesn't run";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- downloads required data.";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- makes dependencies to run project, prepares a model etc.";
	@echo $(YELLOW)"  make compile_cpp "$(NOCOLOR)"- builds the run_cpp executable example";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content"


clean: 
	@echo $(YELLOW)"\nsimple_classifier_cpp: Cleaning up files..."$(NOCOLOR)
	rm -rf build
	rm -rf ${GOOGLENET_BASE_NAME}.xml
	rm -rf ${GOOGLENET_BASE_NAME}.bin
