NCCOMPILE = mo.py

GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'

DATA_RELATIVE_DIR = ../../data
TEST_IMAGE_RELATIVE_DIR = .
MODEL_RELATIVE_DIR = ../../caffe/SSD_Mobilenet
MEAN_VALUES = [127.5,127.5,127.5]
SCALE_VALUES = [127.50223128904757]

INPUT = ../../data/images/nps_chair.png

ZOO_RELATIVE_DIR = ../../omz
MODEL_DOWNLOADER_DIR = open_model_zoo/tools/downloader
MODEL_DOWNLOADER_FILENAME = downloader.py
SSD_MODEL_ZOO_DIR = object_detection/common/mobilenet-ssd/caffe
MODEL_FILE_NAME_BASE = mobilenet-ssd
DEV = MYRIAD

.PHONY: all
all: deps data


.PHONY: deps
deps: compile_model
	@echo $(YELLOW)"\nssd_mobilenet: Making dependencies..."$(NOCOLOR)
	

.PHONY: data
data:
	@echo $(YELLOW)"\nssd_mobilenet: Downloading required data..."$(NOCOLOR)


.PHONY: model_zoo
model_zoo:
	@echo $(YELLOW)"\nssd_mobilenet: Cloning model zoo..."$(NOCOLOR)
	(cd ${ZOO_RELATIVE_DIR}; make all;) 


.PHONY: download_model_files
download_model_files: model_zoo
	@echo $(YELLOW)"\nssd_mobilenet: Checking zoo model files..."$(NOCOLOR)
	@if [ -e ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${SSD_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.caffemodel ] || [ -e ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${SSD_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.prototxt ] ;\
	then \
		echo " - Model files already exists." ; \
	else \
		echo " - Model files do not exist. Using Model downloader to download the model..." ; \
		(cd ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}; python3 ${MODEL_DOWNLOADER_FILENAME} --name ${MODEL_FILE_NAME_BASE};); \
	fi


.PHONY: compile_model
compile_model: download_model_files
	@echo $(YELLOW)"\nssd_mobilenet: Compiling model to IR..."$(NOCOLOR)
	@if [ -e ${MODEL_FILE_NAME_BASE}.xml ] && [ -e ${MODEL_FILE_NAME_BASE}.bin ]; \
	then \
		echo " - IR files already exist. Skipping compile step."; \
	else \
		${NCCOMPILE} --data_type=FP16 --input_model=${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${SSD_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.caffemodel --input_proto=${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${SSD_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.prototxt --mean_values=${MEAN_VALUES} --scale_values=${SCALE_VALUES} || (echo $(RED)"Make sure your paths are correct and the OpenVINO environment variables are set using the "$(YELLOW)"setupvars.sh"$(RED)" script found in <your OpenVINO install location>/bin/ folder."$(NOCOLOR);  exit 1;); \
	fi 

		
.PHONY: run
run: run_py


.PHONY: run_py
run_py: deps data
	@echo $(YELLOW)"\nssd_mobilenet: Running Python sample..."$(NOCOLOR)
	python3 ssd_mobilenet.py --ir=${MODEL_FILE_NAME_BASE}.xml --input=${INPUT} -d ${DEV};
	

.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\nssd_mobilenet: Checking application requirements...\n"$(NOCOLOR)
	@echo "No requirements needed."


.PHONY: uninstall-reqs
uninstall-reqs: 
	@echo $(YELLOW)"\nssd_mobilenet: Uninstalling requirements..."$(NOCOLOR)
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
	

clean: 
	@echo $(YELLOW)"\nssd_mobilenet: Cleaning up files..."$(NOCOLOR)
	@rm -f image.jpg
	rm -f ${MODEL_FILE_NAME_BASE}.bin
	rm -f ${MODEL_FILE_NAME_BASE}.xml
	rm -f ${MODEL_FILE_NAME_BASE}.mapping
	
