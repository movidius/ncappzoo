
# filenames for the graph files that we'll copy to this directory.
CLASSIFIER_GRAPH_BASE_FILENAME = googlenet-v1
CLASSIFIER_RELATIVE_DIR = ../../caffe/GoogLeNet
XML = ${CLASSIFIER_GRAPH_BASE_FILENAME}.xml
BIN = ${CLASSIFIER_GRAPH_BASE_FILENAME}.bin
DEV = MYRIAD

# name of the directory and the base name of the main python file (minus the .py extension)
APP_NAME = benchmark_ncs
GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'


.PHONY: all
all: data deps

.PHONY: googlenet_compile_model
googlenet_compile_model: 
	@echo $(YELLOW)"\nbenchmark_ncs: Compiling the model"$(NOCOLOR)
	@if [ -e ${CLASSIFIER_GRAPH_BASE_FILENAME}.xml ] && [ -e ${CLASSIFIER_GRAPH_BASE_FILENAME}.bin ] ; \
	then \
		echo "${CLASSIFIER_GRAPH_BASE_FILENAME}.xml file already exists, skipping compiliation"; \
	else \
		(cd ${CLASSIFIER_RELATIVE_DIR}; make compile_model; cd ../../apps/${APP_NAME}; cp ${CLASSIFIER_RELATIVE_DIR}/${CLASSIFIER_GRAPH_BASE_FILENAME}.xml . ; cp ${CLASSIFIER_RELATIVE_DIR}/${CLASSIFIER_GRAPH_BASE_FILENAME}.bin . ; ); \
	fi  


.PHONY: data
data: 
	@echo $(YELLOW)"\nbenchmark_ncs: Making the required data..."$(NOCOLOR)


.PHONY: deps
deps: googlenet_compile_model
	@echo $(YELLOW)"\nbenchmark_ncs: Making dependencies..."$(NOCOLOR)


.PHONY: run_py
run_py: install-reqs data deps
	@echo $(YELLOW)"\nbenchmark_ncs: Running the sample..."$(NOCOLOR)
	python3 ./${APP_NAME}.py dev=${DEV} mx=${XML} mb=${BIN}


.PHONY: run
run: run_py


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\nbenchmark_ncs: Checking application requirements..."$(NOCOLOR)
	

.PHONY: help
help:
	@echo "Possible make targets: ";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes everything needed to run but doesn't run";
	@echo $(YELLOW)"  make classifier_model "$(NOCOLOR)"- compiles a classifier model to use when running";
	@echo $(YELLOW)"  make run "$(NOCOLOR)"- runs the application";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements on your system."
	@echo "                      Will remove pip3 opencv and build from source and install a new version." ;
	@echo "                      Only needed once on your system and only if the its not already installed.";
	@echo "                      This may take a long time depending on your system.";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content"

.PHONY: clean
clean: 
	@echo $(YELLOW)"\nbenchmark_ncs: Cleaning up files...""$(NOCOLOR)"
	rm -f ${CLASSIFIER_GRAPH_BASE_FILENAME}.*



