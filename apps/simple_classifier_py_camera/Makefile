
GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'

# name of the directory and the base name of the main python file (minus the .py extension)
APP_NAME = simple_classifier_py

# filenames for the graph files that we'll copy to this directory.
CLASSIFIER_RELATIVE_DIR = ../../apps/simple_classifier_py_camera
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
deps: compile_default_model
	@echo $(YELLOW)'\n'${APP_NAME}": Making dependencies..."$(NOCOLOR)


.PHONY: data
data:
	@echo $(YELLOW)'\n'${APP_NAME}": Making the required data..."$(NOCOLOR)
	@(cd ../../data/ilsvrc12; make)


.PHONY: compile_default_model
compile_default_model:
	@echo $(YELLOW)'\n'${APP_NAME}": Compiling the model to IR..."$(NOCOLOR)
	@if [ -e ${MODEL_FILE_NAME_BASE}.bin ] && [ -e ${MODEL_FILE_NAME_BASE}.xml ]; \
	then \
		echo "- googlenet IR files found. No need to compile."; \
	else \
		echo "- making googlenet IRs in googlenet project."; \
		(cd ${MODEL_RELATIVE_DIR}; make compile_model;);\
		echo "- copying IRs to project folder..."; \
		(cd ${MODEL_RELATIVE_DIR}; cp ${MODEL_FILE_NAME_BASE}.bin ${CLASSIFIER_RELATIVE_DIR}; cp ${MODEL_FILE_NAME_BASE}.xml ${CLASSIFIER_RELATIVE_DIR};); \
	fi 


.PHONY: default_model
default_model: compile_default_model
	@echo $(YELLOW)'\n'${APP_NAME}": Making default models..."$(NOCOLOR)


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\n"$(APP_NAME)": Checking installation requirements..."$(NOCOLOR)
	@echo "No requirements needed."
	

.PHONY: run
run: run_py


.PHONY: run_py
run_py: install-reqs deps data
	@echo $(YELLOW)'\n'${APP_NAME}": Running the sample..."$(NOCOLOR)
	python3 run.py --ir=${MODEL_FILE_NAME_BASE}.xml --labels=${LABELS_RELATIVE_DIR}/${LABELS_FILENAME} 


.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run or run_py"$(NOCOLOR)"- runs the application";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes everything needed to run but doesn't run";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- downloads data as needed";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- makes/prepares dependencies";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make default_model "$(NOCOLOR)"- compiles a default model to use when running";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content";
	@echo ""


clean: 
	@echo $(YELLOW)'\n'${APP_NAME}": Cleaning up files..."$(NOCOLOR)
	rm -f ${MODEL_FILE_NAME_BASE}.xml
	rm -f ${MODEL_FILE_NAME_BASE}.bin
