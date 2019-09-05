NCCOMPILE = mo.py

YELLOW='\033[1;33m'
NOCOLOR='\033[0m'
RED = '\033[1;31m'

MEAN_VALUES = [127.5,127.5,127.5]
SCALE = 128
MODEL_FILE_NAME_BASE = 20180408-102900

ZOO_RELATIVE_DIR = ../../omz
MODEL_DOWNLOADER_DIR = open_model_zoo/tools/downloader
MODEL_DOWNLOADER_FILENAME = downloader.py
FACENET_MODEL_ZOO_DIR = face_recognition/facenet/CASIA-WebFace/tf/20180408-102900
DOWNLOADER_MODEL_NAME = facenet-20180408-102900


GET_PICTURES = wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics_cropped/licenses.txt; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics_cropped/elvis.png; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics_cropped/trump.png; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics_cropped/reagan.png


.PHONY: all
all: deps data compile_model


.PHONY: deps
deps: download_model_files
	@echo $(YELLOW)"\nfacenet: Making dependencies..."$(NOCOLOR)


.PHONY: data
data: pictures
	@echo $(YELLOW)"\nfacenet: Downloading required data for model..."$(NOCOLOR)


.PHONY: pictures
pictures:
	@echo "\nfacenet: Downloading images..."
	(cd test_faces; ${GET_PICTURES};)

.PHONY: model_zoo
model_zoo:
	@echo $(YELLOW)"\nfacenet: Making model zoo..."$(NOCOLOR)
	(cd ${ZOO_RELATIVE_DIR}; make all;) 


.PHONY: download_model_files
download_model_files: model_zoo	
	@echo $(YELLOW)"\nfacenet: Checking zoo model files..."$(NOCOLOR)
	@if [ -e ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${FACENET_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.pb ] ;\
	then \
		echo " - Model files already exists." ; \
	else \
		echo " - Model files do not exist. Using Model downloader to download the model..." ; \
		(cd ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}; python3 ${MODEL_DOWNLOADER_FILENAME} --name ${DOWNLOADER_MODEL_NAME};); \
	fi


.PHONY: compile_model
compile_model: download_model_files 
	@echo $(YELLOW)"\nfacenet: Compiling model to IR..."$(NOCOLOR)
	@if [ -z "$(INTEL_OPENVINO_DIR)" ] ; \
	then \
		echo "Please initiate the Intel OpenVINO environment by going to the installation directory for openvino and running the setupvars.sh file in the bin folder." ; \
		exit 1 ; \
	else \
		echo "Intel OpenVINO environment is already set!" ; \
	fi
	@if [ -e ${MODEL_FILE_NAME_BASE}.xml ] && [ -e ${MODEL_FILE_NAME_BASE}.bin ] ; \
	then \
		echo " - Compiled model file already exists, skipping compile."; \
	else \
		${NCCOMPILE} --data_type=FP16 --reverse_input_channels --framework=tf --freeze_placeholder_with_value="phase_train->False" --input_shape=[1,160,160,3],[1] --input=image_batch,phase_train --output=embeddings --scale=${SCALE} --mean_values=${MEAN_VALUES} --input_model=${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${FACENET_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.pb; \
	fi;


.PHONY: run
run: run_py


.PHONY: run_py
run_py: data deps compile_model
	@echo $(YELLOW) "\nfacenet: Running Python sample..." $(NOCOLOR)
	python3 facenet.py;


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\nfacenet: Checking application requirements...\n"$(NOCOLOR)
	@echo "No requirements needed."


.PHONY: uninstall-reqs
uninstall-reqs: 
	@echo $(YELLOW)"\nfacenet: Uninstalling requirements..."$(NOCOLOR)
	@echo "Nothing to uninstall."
		

.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run or run_py "$(NOCOLOR)"- Runs Python example.";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- Shows this message.";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- Makes everything needed to run, but doesn't run.";
	@echo $(YELLOW)"  make compile_model "$(NOCOLOR)"- Runs model compiler for the network.";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- Downloads required data.";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- Makes dependencies for project, prepares model etc.";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- Removes files created in this directory.";
	@echo "  "

.PHONY: clean
clean:
	@echo $(YELLOW)"\nfacenet: Cleaning up files..."$(NOCOLOR);
	rm -f test_faces/licenses.txt
	rm -f test_faces/trump.png
	rm -f test_faces/elvis.png
	rm -f test_faces/reagan.png
	rm -f ${MODEL_FILE_NAME_BASE}.xml
	rm -f ${MODEL_FILE_NAME_BASE}.bin
	rm -f ${MODEL_FILE_NAME_BASE}.mapping
	
	
