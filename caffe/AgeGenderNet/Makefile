NCCOMPILE = mo.py

GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'

DATA_RELATIVE_DIR = ../../data
MODEL_RELATIVE_DIR = ../AgeGenderNet

TEST_IMAGE_FILENAME = image.jpg
CROPPED_IMAGE_FILENAME = cropped_face_0.png

ZOO_RELATIVE_DIR = ../../omz
MODEL_DOWNLOADER_DIR = open_model_zoo/tools/downloader
MODEL_DOWNLOADER_FILENAME = downloader.py

AGE_GENDER_MODEL_ZOO_DIR = Retail/object_attributes/age_gender/dldt/FP16
MODEL_FILE_NAME_BASE = age-gender-recognition-retail-0013

GET_TEST_IMAGE = wget -O image.jpg -N http://vis-www.cs.umass.edu/lfw/images/Talisa_Bratt/Talisa_Bratt_0001.jpg

FACE_DETECT_RELATIVE_DIR = ../face-detection-retail-0004

.PHONY: all
all: deps data 


.PHONY: deps
deps: get_ir
	@echo $(YELLOW)"\nAgeGenderNet: Making dependencies..."$(NOCOLOR)
	
	
.PHONY: data
data: face_detect_compile
	@echo $(YELLOW)"\nAgeGenderNet: Downloading required data..."$(NOCOLOR)
	test -f ${TEST_IMAGE_FILENAME} || ${GET_TEST_IMAGE}


.PHONY: model_zoo
model_zoo:
	@echo $(YELLOW)"\nAgeGenderNet: Making model zoo..."$(NOCOLOR)
	(cd ${ZOO_RELATIVE_DIR}; make all;) 


.PHONY: get_ir
get_ir:	model_zoo
	@echo $(YELLOW)"\nAgeGenderNet: Downloading IRs..."$(NOCOLOR)
	@if [ -e ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${AGE_GENDER_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.xml ] || [ -e ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${AGE_GENDER_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.bin ]; \
	then \
		echo " - Model files already exists in the open model zoo."; \
	else \
		echo " - Model files do not exist in the open model zoo. ";\
		echo " - Using Model downloader to download the model..."; \
		(cd ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}; python3 ${MODEL_DOWNLOADER_FILENAME} --name ${MODEL_FILE_NAME_BASE};); \
	fi; \
	
	@if [ -e ${MODEL_FILE_NAME_BASE}.xml ] && [ -e ${MODEL_FILE_NAME_BASE}.bin ]; \
	then \
		echo " - IR files already exist in project directory. No need to copy IRs." ;\
	else \
		echo " - IR files do not exist in project directory.";\
		echo " - Copying IR files to current folder..." ;\
		cp ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${AGE_GENDER_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.xml . ;\
		cp ${ZOO_RELATIVE_DIR}/${MODEL_DOWNLOADER_DIR}/${AGE_GENDER_MODEL_ZOO_DIR}/${MODEL_FILE_NAME_BASE}.bin . ;\
	fi
		
		
.PHONY: face_detect_compile
face_detect_compile:
	@echo $(YELLOW)"\nAgeGenderNet: Compiling face detection IR..."$(NOCOLOR)
	(cd ${FACE_DETECT_RELATIVE_DIR}; make all;)


.PHONY: run_face_detect
run_face_detect:
	@echo $(YELLOW)"\nAgeGenderNet: Running face detection sample to get cropped face mat..."$(NOCOLOR)
	(cd ${FACE_DETECT_RELATIVE_DIR}; make run SHOW="--show=no" CROP="--crop=" CROPPED_FACES_DIR=${MODEL_RELATIVE_DIR};)


.PHONY: compile_model
compile_model: 
	@echo $(YELLOW)"\nAgeGenderNet: Compiling model to IR..."$(NOCOLOR)
	@echo " - No compiling needed. Using Pre-compiled IRs from the open model zoo."


.PHONY: run
run: run_py


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\nAgeGenderNet: Checking application requirements...\n"$(NOCOLOR)
	@echo "No requirements needed."
	

.PHONY: run_py
run_py: install-reqs deps data compile_model run_face_detect
	@echo $(YELLOW)"\nAgeGenderNet: Running Python sample..."$(NOCOLOR)
	python3 age_gender_net.py --ir=${MODEL_FILE_NAME_BASE}.xml --image=${CROPPED_IMAGE_FILENAME};


.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run "$(NOCOLOR)"- Runs the Python example.";
	@echo $(YELLOW)"  make run_py "$(NOCOLOR)"- Runs the Python example.";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- Shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- Makes everything needed to run, but doesn't run.";
	@echo $(YELLOW)"  make compile_model "$(NOCOLOR)"- Compiles the model to an IR format.";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- Downloads the required data.";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- Makes dependencies for project, downloads and prepares model etc.";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- Removes all temporary files that are created by the Makefile.\n";
	

clean: 
	@echo $(YELLOW)"\nAgeGenderNet: Cleaning up files..."$(NOCOLOR)
	@rm -f ${TEST_IMAGE_FILENAME}
	rm -f ${MODEL_FILE_NAME_BASE}.bin
	rm -f ${MODEL_FILE_NAME_BASE}.xml
	rm -f ${MODEL_FILE_NAME_BASE}.mapping
	@rm -f cropped_face_*.*
    
