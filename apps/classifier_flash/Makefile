
# filenames for the graph files that we'll copy to this directory.
CLASSIFIER_GRAPH_BASE_FILENAME = googlenet-v1
CLASSIFIER_RELATIVE_DIR = ../../caffe/GoogLeNet
XML = ${CLASSIFIER_GRAPH_BASE_FILENAME}.xml
BIN = ${CLASSIFIER_GRAPH_BASE_FILENAME}.bin
DEV = MYRIAD
IMAGES_FOLDER = images

# name of the directory and the base name of the main python file (minus the .py extension)
GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'
APP_NAME = classifier_flash


.PHONY: all
all: data deps
	

.PHONY: data
data: 
	@echo $(YELLOW)'\n'${APP_NAME}": Making the required data..."$(NOCOLOR)
	(cd ../../data/ilsvrc12; make all; cd ../../apps/${APP_NAME};)


.PHONY: deps
deps: default_model
	@echo $(YELLOW)'\n'${APP_NAME}": Making dependencies..."$(NOCOLOR)
	@if [ -d ${IMAGES_FOLDER} ]; \
	then \
		echo "Images folder already exists."; \
	else \
		echo "Copying images from "${IMAGES_FOLDER}"..."; \
		(mkdir -p images; cp ../../data/images/pic*.jpg ./images; rename 's/pic/copy_image_1/' images/*; cp ../../data/images/pic*.jpg ./images; rename 's/pic/copy_image_2/' images/*;) ;\
	fi


.PHONY: compile_default_model
compile_default_model:
	@echo $(YELLOW)'\n'${APP_NAME}": Compiling default model to IR..."$(NOCOLOR)
	@if [ -e ${CLASSIFIER_GRAPH_BASE_FILENAME}.xml ] ; \
	then \
		echo "${CLASSIFIER_GRAPH_BASE_FILENAME}.xml file already exists, skipping compiliation"; \
	else \
		(cd ${CLASSIFIER_RELATIVE_DIR}; make compile_model; cd ../../apps/${APP_NAME}; cp ${CLASSIFIER_RELATIVE_DIR}/${CLASSIFIER_GRAPH_BASE_FILENAME}.xml . ; cp ${CLASSIFIER_RELATIVE_DIR}/${CLASSIFIER_GRAPH_BASE_FILENAME}.bin . ; ); \
	fi  


.PHONY: default_model
default_model: compile_default_model
	@echo $(YELLOW)'\n'${APP_NAME}": Making the default model..."$(NOCOLOR)


.PHONY: run_py
run_py: data deps
	@echo $(YELLOW)'\n'${APP_NAME}": Running the Python sample..."$(NOCOLOR)
	python3 ./${APP_NAME}.py dev=${DEV} mx=${XML} mb=${BIN} 


.PHONY: run
run: run_py


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)'\n'${APP_NAME}": No installation requirements needed."$(NOCOLOR)

	
.PHONY: uninstall-reqs
uninstall-reqs: 
	@echo $(YELLOW)'\n'${APP_NAME}": Uninstalling requirements..."$(NOCOLOR)
	@echo "Nothing to uninstall."
	
	
.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run or run_py"$(NOCOLOR)"- runs the application";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes everything needed to run but doesn't run";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- downloads data as needed";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- makes/prepares dependencies";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make default_model "$(NOCOLOR)"- compiles a default model to use when running";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content";
	@echo ""
	
.PHONY: clean
clean: 
	@echo $(YELLOW)'\n'${APP_NAME}": Cleaning up files..."$(NOCOLOR)
	rm -f ${CLASSIFIER_GRAPH_BASE_FILENAME}.*
	rm -rf images


