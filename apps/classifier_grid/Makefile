
GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'

# filenames for the graph files that we'll copy to this directory.
CLASSIFIER_GRAPH_BASE_FILENAME = googlenet-v1
CLASSIFIER_RELATIVE_DIR = ../../caffe/GoogLeNet
XML = ${CLASSIFIER_GRAPH_BASE_FILENAME}.xml
BIN = ${CLASSIFIER_GRAPH_BASE_FILENAME}.bin
DEV = MYRIAD

# name of the directory and the base name of the main python file (minus the .py extension)
APP_NAME = classifier_grid
IMAGES_FOLDER = images


.PHONY: all
all: data deps

.PHONY: classifier_model
classifier_model: 
	@echo $(YELLOW)"\nclassifier_grid: Making classifier model..."$(NOCOLOR)
	@if [ -e ${CLASSIFIER_GRAPH_BASE_FILENAME}.xml ] ; \
	then \
		echo "${CLASSIFIER_GRAPH_BASE_FILENAME}.xml file already exists, skipping compiliation"; \
	else \
		(cd ${CLASSIFIER_RELATIVE_DIR}; make compile_model; cd ../../apps/${APP_NAME}; cp ${CLASSIFIER_RELATIVE_DIR}/${CLASSIFIER_GRAPH_BASE_FILENAME}.xml . ; cp ${CLASSIFIER_RELATIVE_DIR}/${CLASSIFIER_GRAPH_BASE_FILENAME}.bin . ; ); \
	fi  
	

.PHONY: data
data: 
	@echo $(YELLOW)"\nclassifier_grid: Making the required data..."$(NOCOLOR)
	(cd ../../data/ilsvrc12; make all; cd ../../apps/${APP_NAME};)

.PHONY: deps
deps: classifier_model
	@echo $(YELLOW)"\nclassifier_grid: Making dependencies..."$(NOCOLOR)
	@if [ -d ${IMAGES_FOLDER} ]; \
	then \
		echo "Images folder already exists."; \
	else \
		echo "Copying images from "${IMAGES_FOLDER}"..."; \
		(mkdir -p images; cp ../../data/images/pic*.jpg ./images; rename 's/pic/copy_image_1/' images/*; cp ../../data/images/pic*.jpg ./images; rename 's/pic/copy_image_2/' images/*; cp ../../data/images/pic*.jpg ./images; rename 's/pic/copy_image_3/' images/*; cp ../../data/images/pic*.jpg ./images; rename 's/pic/copy_image_4/' images/*; cp ../../data/images/pic*.jpg ./images; rename 's/pic/copy_image_5/' images/*; cp ../../data/images/pic*.jpg ./images; rename 's/pic/copy_image_6/' images/*; cp ../../data/images/pic*.jpg ./images; rename 's/pic/copy_image_7/' images/*; cp ../../data/images/pic*.jpg ./images; rename 's/pic/copy_image_8/' images/*; cp ../../data/images/pic*.jpg ./images; rename 's/pic/copy_image_9/' images/*; cp ../../data/images/pic*.jpg ./images; rename 's/pic/copy_image_10/' images/*;) ;\
	fi

.PHONY: run_py
run_py: install-reqs data deps
	@echo $(YELLOW)"\nclassifier_grid: Running the Python sample..."$(NOCOLOR)
	python3 ./${APP_NAME}.py dev=${DEV} mx=${XML} mb=${BIN}


.PHONY: run
run: run_py


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\nclassifier_grid: Checking application requirements..."$(NOCOLOR)
	

.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes everything needed to run but doesn't run";
	@echo $(YELLOW)"  make classifier_model "$(NOCOLOR)"- compiles a classifier model to use when running";
	@echo $(YELLOW)"  make run "$(NOCOLOR)"- runs the application";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements on your system."
	@echo "                      Will remove pip3 opencv and build from source and install a new version." ;
	@echo "                      Only needed once on your system and only if the its not already installed.";
	@echo "                      This may take a long time depending on your system.";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content"
	@echo ""

.PHONY: clean
clean: 
	@echo $(YELLOW)"\nclassifier_grid: Cleaning up files..."$(NOCOLOR)
	rm -f ${CLASSIFIER_GRAPH_BASE_FILENAME}.*
	rm -rf ${IMAGES_FOLDER}



