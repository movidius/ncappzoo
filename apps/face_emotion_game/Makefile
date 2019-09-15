NCCOMPILE = mo.py

GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'

ZOO_RELATIVE_DIR = ../../omz
MODEL_DOWNLOADER_DIR = open_model_zoo/tools/downloader
MODEL_DOWNLOADER_FILENAME = downloader.py
FACE_MODEL_ZOO_DIR = Retail/object_detection/face/sqnet1.0modif-ssd/0004/dldt/FP16
EMOTION_MODEL_ZOO_DIR = Retail/object_attributes/emotions_recognition/0003/dldt/INT8
FACE_MODEL_FILE_NAME_BASE = face-detection-retail-0004
EMOTION_MODEL_FILE_NAME_BASE = emotions-recognition-retail-0003
FACE_DESTINATION_DIR = src/data/face-detection
EMOTION_DESTINATION_DIR = src/data/emotions-recognition

FACE_NETWORK_NAME = face_detection_retail_0004
EMOTION_NETWORK_NAME = emotions_recognition_retail_0003
GAME_NAME = face_emotion_game
CROP = 
CROPPED_FACES_DIR = 
SHOW = --show=yes

.PHONY: all
all: deps data


.PHONY: deps
deps: get_ir
	@echo $(YELLOW)'\n'${FACE_NETWORK_NAME}": Making dependencies..."$(NOCOLOR)
	

.PHONY: data
data:
	@echo $(YELLOW)'\n'${FACE_NETWORK_NAME}": Downloading required data..."$(NOCOLOR)


.PHONY: model_zoo
model_zoo:
	@echo $(YELLOW)'\n'${FACE_NETWORK_NAME}": Cloning model zoo..."$(NOCOLOR)
	(cd ${ZOO_RELATIVE_DIR}; make all;) 


.PHONY: get_ir
get_ir:	model_zoo
	@echo $(YELLOW)'\n'${FACE_NETWORK_NAME}": Downloading IRs..."$(NOCOLOR);
	@if [ -e ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${FACE_MODEL_ZOO_DIR}/${FACE_MODEL_FILE_NAME_BASE}.xml ] || [ -e ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${FACE_MODEL_ZOO_DIR}/${FACE_MODEL_FILE_NAME_BASE}.bin ]; \
	then \
		echo " - Model files already exists in open model directory."; \
	else \
		echo " - Model files do not exist in open model directory. ";\
		echo " - Using Model downloader to download the model..."; \
		(cd ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}; python3 ${MODEL_DOWNLOADER_FILENAME} --name ${FACE_MODEL_FILE_NAME_BASE};); \
	fi
	
	@if [ -e ${FACE_MODEL_FILE_NAME_BASE}.xml ] && [ -e ${FACE_MODEL_FILE_NAME_BASE}.bin ]; \
	then \
		echo " - IR files already exist in project directory. No need to copy IRs."; \
	else \
		echo " - IR files do not exist in project directory.";\
		echo " - Copying IR files to current folder..."; \
		cp ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${FACE_MODEL_ZOO_DIR}/${FACE_MODEL_FILE_NAME_BASE}.xml ${FACE_DESTINATION_DIR}; \
		cp ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${FACE_MODEL_ZOO_DIR}/${FACE_MODEL_FILE_NAME_BASE}.bin ${FACE_DESTINATION_DIR}; \
	fi
	
    @echo $(YELLOW)'\n'${EMOTION_NETWORK_NAME}": Downloading IRs..."$(NOCOLOR);
	@if [ -e ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${EMOTION_MODEL_ZOO_DIR}/${EMOTION_MODEL_FILE_NAME_BASE}.xml ] || [ -e ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${EMOTION_MODEL_ZOO_DIR}/${EMOTION_MODEL_FILE_NAME_BASE}.bin ]; \
	then \
		echo " - Model files already exists in open model directory."; \
	else \
		echo " - Model files do not exist in open model directory. ";\
		echo " - Using Model downloader to download the model..."; \
		(cd ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}; python3 ${MODEL_DOWNLOADER_FILENAME} --name ${EMOTION_MODEL_FILE_NAME_BASE};); \
	fi

	@if [ -e ${FACE_MODEL_FILE_NAME_BASE}.xml ] && [ -e ${FACE_MODEL_FILE_NAME_BASE}.bin ]; \
	then \
		echo " - IR files already exist in project directory. No need to copy IRs."; \
	else \
		echo " - IR files do not exist in project directory.";\
		echo " - Copying IR files to current folder..."; \
		cp ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${EMOTION_MODEL_ZOO_DIR}/${EMOTION_MODEL_FILE_NAME_BASE}.xml ${EMOTION_DESTINATION_DIR}; \
		cp ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${EMOTION_MODEL_ZOO_DIR}/${EMOTION_MODEL_FILE_NAME_BASE}.bin ${EMOTION_DESTINATION_DIR}; \
	fi


.PHONY: compile_model
compile_model: 
	@echo $(YELLOW)'\n'${FACE_NETWORK_NAME}": Compiling model to IR..."$(NOCOLOR)
	@echo " - No compiling needed. Using Pre-compiled IRs from the open model zoo."

	
.PHONY: run
run: run_py


.PHONY: run_py
run_py: deps data
	@echo $(YELLOW)'\n'${GAME_NAME}": Running Python sample..."$(NOCOLOR)
	@echo "Checking OpenVINO environment..."
	@if [ -z "$(INTEL_OPENVINO_DIR)" ] ; \
	then \
		echo "Please initiate the Intel OpenVINO environment by going to the installation directory for openvino and running the setupvars.sh file in the bin folder." ; \
		exit 1 ; \
	else \
		echo "Intel OpenVINO environment is already set!" ; \
	fi
	python3 ${GAME_NAME}.py -fps -m


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)'\n'${FACE_NETWORK_NAME}": Checking installation requirements..."$(NOCOLOR)
	@echo "No requirements needed."


.PHONY: uninstall-reqs
uninstall-reqs: 
	@echo $(YELLOW)'\n'${FACE_NETWORK_NAME}": Uninstalling requirements..."$(NOCOLOR)
	@echo "Nothing to uninstall."
	

.PHONY: help
help:
	@echo "\n Possible make targets: ";
	@echo $(YELLOW)"  make run or make run_py "$(NOCOLOR)"- Runs the Python example.";
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
	@echo "  This network does face detection and can be used to align faces.";
	@echo $(YELLOW)" Model inputs:"$(NOCOLOR)
	@echo "  name: 'input', shape: [1x3x300x300], Expected color order is BGR.";
	@echo $(YELLOW)" Model outputs:"$(NOCOLOR)
	@echo "  name: 'detection_out', shape: [1, 1, N, 7] - where N is the number of detected bounding boxes. For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max].\n";

clean: 
	@echo $(YELLOW)'\n'${FACE_NETWORK_NAME}": Cleaning up files..."$(NOCOLOR)
	@rm -f image.jpg
	rm -f ${MODEL_FILE_NAME_BASE}.bin
	rm -f ${MODEL_FILE_NAME_BASE}.xml
	rm -f ${MODEL_FILE_NAME_BASE}.mapping
