NCCOMPILE = mo.py

GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'

DATA_RELATIVE_DIR = ../../data
MEAN_VALUES = [127.5,127.5,127.5]
SCALE_VALUES = [127.50223128904757]

INPUT = ../../data/images/nps_chair.png

ZOO_RELATIVE_DIR = ../../omz
MODEL_DOWNLOADER_DIR = open_model_zoo/tools/downloader
MODEL_DOWNLOADER_FILENAME = downloader.py
SSD_MODEL_ZOO_DIR = object_detection/common/mobilenet-ssd/caffe
MODEL_FILE_NAME_BASE = mobilenet-ssd
DEV = MYRIAD

NETWORK_NAME = ssd_mobilenet_v1_caffe
IR = ${MODEL_FILE_NAME_BASE}.xml
LABELS = labels.txt

.PHONY: all
all: deps data


.PHONY: deps
deps: compile_model
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Making dependencies..."$(NOCOLOR)
	

.PHONY: data
data:
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Downloading required data..."$(NOCOLOR)


.PHONY: model_zoo
model_zoo:
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Cloning model zoo..."$(NOCOLOR)
	(cd ${ZOO_RELATIVE_DIR}; make all;) 


.PHONY: download_model_files
download_model_files: model_zoo
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Checking zoo model files..."$(NOCOLOR)
	@if [ -e ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${SSD_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.caffemodel ] || [ -e ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${SSD_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.prototxt ] ;\
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
		${NCCOMPILE} --data_type=FP16 --input_model=${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${SSD_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.caffemodel --input_proto=${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${SSD_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.prototxt --mean_values=${MEAN_VALUES} --scale_values=${SCALE_VALUES}; \
	fi 

		
.PHONY: run
run: run_py


.PHONY: run_py
run_py: deps data
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Running Python sample..."$(NOCOLOR)
	python3 ${NETWORK_NAME}.py --ir=${IR} --input=${INPUT} -d ${DEV} --labels=${LABELS};
	

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
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- Shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- Makes everything needed to run, but doesn't run.";
	@echo $(YELLOW)"  make compile_model "$(NOCOLOR)"- Compiles the model to an IR format.";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- Downloads the required data.";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- Makes dependencies for project, downloads and prepares model etc.";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- Removes all temporary files that are created by the Makefile.\n";
	@echo " ----- "$(YELLOW)"Model information"$(NOCOLOR)" -----"
	@echo $(YELLOW)" Description: "$(NOCOLOR)
	@echo "  This network does object detection on 20 different classes.";
	@echo $(YELLOW)" Model inputs:"$(NOCOLOR)
	@echo "  name: 'data', shape: [1x3x300x300], Expected color order is BGR.";
	@echo $(YELLOW)" Model outputs:"$(NOCOLOR)
	@echo "  name: 'detection_out', shape: [1, 1, N, 7] - where N is the number of detected bounding boxes. For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max].\n";
	

clean: 
	@echo $(YELLOW)'\n'${NETWORK_NAME}": Cleaning up files..."$(NOCOLOR)
	@rm -f image.jpg
	rm -f ${MODEL_FILE_NAME_BASE}.bin
	rm -f ${MODEL_FILE_NAME_BASE}.xml
	rm -f ${MODEL_FILE_NAME_BASE}.mapping
	
