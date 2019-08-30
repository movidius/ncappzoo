
GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'

# name of the directory and the base name of the main python file (minus the .py extension)
APP_NAME = simple_classifier_cpp

# filenames for the graph files that we'll copy to this directory.
CLASSIFIER_RELATIVE_DIR = ../../apps/simple_classifier_cpp
GOOGLENET_RELATIVE_DIR = ../../caffe/GoogLeNet
GOOGLENET_BASE_NAME = googlenet-v1

IMAGES_RELATIVE_DIR = ../../data/images
TEST_IMAGE_FILENAME = nps_electric_guitar.png

LABELS_RELATIVE_DIR = ../../data/ilsvrc12
LABELS_FILENAME = synset_labels.txt


.PHONY: all
all: deps data compile_cpp


.PHONY: data
data: 
	@echo $(YELLOW)'\n'${APP_NAME}": Downloading the required data..."$(NOCOLOR)
	(cd ${LABELS_RELATIVE_DIR}; make;)


.PHONY: deps
deps: default_model
	@echo $(YELLOW)'\n'${APP_NAME}": Making dependencies..."$(NOCOLOR)


.PHONY: compile_default_model
compile_default_model:
	@echo $(YELLOW)'\n'${APP_NAME}": Compiling default model to IR..."$(NOCOLOR)
	@if [ -e ${GOOGLENET_BASE_NAME}.bin ] && [ -e ${GOOGLENET_BASE_NAME}.xml ]; \
	then \
		echo " - Googlenet IR files found. No need to compile."; \
	else \
		echo " - Making googlenet IRs in googlenet project."; \
		(cd ${GOOGLENET_RELATIVE_DIR}; make deps; make data; make compile_model;);\
		echo " - Copying IR files to project directory"; \
		(cd ${GOOGLENET_RELATIVE_DIR}; cp ${GOOGLENET_BASE_NAME}.bin ${CLASSIFIER_RELATIVE_DIR}; cp ${GOOGLENET_BASE_NAME}.xml ${CLASSIFIER_RELATIVE_DIR};); \
	fi 
	
	
.PHONY: default_model
default_model: compile_default_model
	@echo $(YELLOW)'\n'${APP_NAME}": Making default models..."$(NOCOLOR)


.PHONY: run
run: run_cpp
	@echo $(YELLOW)'\n'${APP_NAME}": Running the sample..."$(NOCOLOR)


.PHONY:	compile_cpp
compile_cpp:  
	@echo $(YELLOW)'\n'${APP_NAME}": Building the sample..."$(NOCOLOR)
	(mkdir -p build; cd build; cmake ..; make)


.PHONY: run_cpp
run_cpp: deps data compile_cpp
	@echo $(YELLOW)'\n'${APP_NAME}": Running the sample..."$(NOCOLOR)
	./build/${APP_NAME} ${GOOGLENET_BASE_NAME}.xml ${IMAGES_RELATIVE_DIR}/${TEST_IMAGE_FILENAME} ${LABELS_RELATIVE_DIR}/${LABELS_FILENAME}


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\n"$(APP_NAME)": Checking installation requirements..."$(NOCOLOR)
	@echo "No requirements needed."	


.PHONY: uninstall-reqs
uninstall-reqs: 
	@echo $(YELLOW)'\n'${APP_NAME}": Uninstalling requirements..."$(NOCOLOR)
	@echo "Nothing to uninstall."
	
.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run or run_cpp"$(NOCOLOR)"- runs the application";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes everything needed to run but doesn't run";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- downloads data as needed";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- makes/prepares dependencies";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make default_model "$(NOCOLOR)"- compiles a default model to use when running";
	@echo $(YELLOW)"  make compile_cpp "$(NOCOLOR)"- builds the run_cpp executable example";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content";
	@echo ""


clean: 
	@echo $(YELLOW)'\n'${APP_NAME}": Cleaning up files..."$(NOCOLOR)
	rm -rf build
	rm -rf ${GOOGLENET_BASE_NAME}.xml
	rm -rf ${GOOGLENET_BASE_NAME}.bin
