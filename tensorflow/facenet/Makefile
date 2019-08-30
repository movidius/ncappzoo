NCCOMPILE = mo.py

YELLOW='\033[1;33m'
NOCOLOR='\033[0m'
RED = '\033[1;31m'

PROJECT_DIR = $(CURDIR)
ZOO_TF_SRC_PATH = ${PROJECT_DIR}/${ZOO_RELATIVE_TF_SRC_PROJ}/tensorflow
ZOO_TF_MODELS_PATH = ${PROJECT_DIR}/${ZOO_RELATIVE_TF_MODELS_PROJ}/models

MEAN_VALUES = [127.5,127.5,127.5]
SCALE = 128
MODEL_FILE_NAME_BASE = 20180408-102900

ZOO_RELATIVE_DIR = ../../omz
MODEL_DOWNLOADER_DIR = open_model_zoo/tools/downloader
MODEL_DOWNLOADER_FILENAME = downloader.py
FACENET_MODEL_ZOO_DIR = face_recognition/facenet/CASIA-WebFace/tf/20180408-102900
DOWNLOADER_MODEL_NAME = facenet-20180408-102900

FACE_DETECT_RELATIVE_DIR = ../../caffe/face-detection-retail-0004
MODEL_RELATIVE_DIR = ../../tensorflow/facenet

GET_PICTURES = wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics/licenses.txt; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics/elvis-presley-401920_640.jpg; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics/trump.jpg; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics/president-67550_640.jpg

.PHONY: all
all: deps data compile_model


.PHONY: deps
deps: download_model_files
	@echo $(YELLOW)"\nfacenet: Making dependencies..."$(NOCOLOR)


.PHONY: data
data: pictures
	mkdir -p validated_faces
	mkdir -p test_faces
	@echo $(YELLOW)"\nfacenet: Downloading required data for model..."$(NOCOLOR)


.PHONY: pictures
pictures:
	@echo "\nfacenet: Downloading images..."
	${GET_PICTURES};

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
	@if [ -e ${MODEL_FILE_NAME_BASE}.xml ] && [ -e ${MODEL_FILE_NAME_BASE}.bin ] ; \
	then \
		echo " - Compiled model file already exists, skipping compile."; \
	else \
		${NCCOMPILE} --data_type=FP16 --reverse_input_channels --framework=tf --freeze_placeholder_with_value="phase_train->False" --input_shape=[1,160,160,3],[1] --input=image_batch,phase_train --output=embeddings --scale=${SCALE} --mean_values=${MEAN_VALUES} --input_model=${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${FACENET_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.pb; \
	fi;


.PHONY: run_face_detect
run_face_detect: pictures
	@echo $(YELLOW)"\nfacenet: Running face detection sample to get cropped face mat..."$(NOCOLOR)
	@if [ ! -e validated_faces/valid_face.png ]; then \
		(cd ${FACE_DETECT_RELATIVE_DIR}; make run SHOW="--show=no" CROP="--crop=" CROPPED_FACES_DIR=${MODEL_RELATIVE_DIR}/validated_faces TEST_IMAGE_FILENAME=${MODEL_RELATIVE_DIR}/valid.jpg; mv ${MODEL_RELATIVE_DIR}/validated_faces/cropped_face_0.png ${MODEL_RELATIVE_DIR}/validated_faces/valid_face.png;); \
	fi
	@if [ ! -e test_faces/valid_face.png ]; then \
		(cd ${FACE_DETECT_RELATIVE_DIR}; make run SHOW="--show=no" CROP="--crop=" CROPPED_FACES_DIR=${MODEL_RELATIVE_DIR}/test_faces TEST_IMAGE_FILENAME=${MODEL_RELATIVE_DIR}/valid.jpg; mv ${MODEL_RELATIVE_DIR}/test_faces/cropped_face_0.png ${MODEL_RELATIVE_DIR}/test_faces/valid_face.png;); \
	fi
	@if [ ! -e test_faces/elvis.png ]; then \
		(cd ${FACE_DETECT_RELATIVE_DIR}; make run SHOW="--show=no" CROP="--crop=" CROPPED_FACES_DIR=${MODEL_RELATIVE_DIR}/test_faces TEST_IMAGE_FILENAME=${MODEL_RELATIVE_DIR}/elvis-presley-401920_640.jpg; mv ${MODEL_RELATIVE_DIR}/test_faces/cropped_face_0.png ${MODEL_RELATIVE_DIR}/test_faces/elvis.png;); \
	fi
	@if [ ! -e test_faces/trump.png ]; then \
		(cd ${FACE_DETECT_RELATIVE_DIR}; make run SHOW="--show=no" CROP="--crop=" CROPPED_FACES_DIR=${MODEL_RELATIVE_DIR}/test_faces TEST_IMAGE_FILENAME=${MODEL_RELATIVE_DIR}/trump.jpg; mv ${MODEL_RELATIVE_DIR}/test_faces/cropped_face_0.png ${MODEL_RELATIVE_DIR}/test_faces/trump.png;); \
	fi
	@if [ ! -e test_faces/reagan.png ]; then \
		(cd ${FACE_DETECT_RELATIVE_DIR}; make run SHOW="--show=no" CROP="--crop=" CROPPED_FACES_DIR=${MODEL_RELATIVE_DIR}/test_faces TEST_IMAGE_FILENAME=${MODEL_RELATIVE_DIR}/president-67550_640.jpg; mv ${MODEL_RELATIVE_DIR}/test_faces/cropped_face_0.png ${MODEL_RELATIVE_DIR}/test_faces/reagan.png;); \
	fi
	@if [ ! -e test_faces/nealvis.png ]; then \
		(cd ${FACE_DETECT_RELATIVE_DIR}; make run SHOW="--show=no" CROP="--crop=" CROPPED_FACES_DIR=${MODEL_RELATIVE_DIR}/test_faces TEST_IMAGE_FILENAME=${MODEL_RELATIVE_DIR}/neal_2017-12-19-155037.jpg; mv ${MODEL_RELATIVE_DIR}/test_faces/cropped_face_0.png ${MODEL_RELATIVE_DIR}/test_faces/nealvis.png;); \
	fi


.PHONY: run
run: run_py


.PHONY: run_py
run_py: data deps compile_model run_face_detect
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
	rm -rf test_faces
	rm -rf validated_faces
	rm -f licenses.txt
	rm -f trump.jpg
	rm -f elvis-presley-401920_640.jpg
	rm -f president-67550_640.jpg
	rm -f ${MODEL_FILE_NAME_BASE}.xml
	rm -f ${MODEL_FILE_NAME_BASE}.bin
	rm -f ${MODEL_FILE_NAME_BASE}.mapping
	
	
