NCCOMPILE = mo.py

YELLOW='\033[1;33m'
NOCOLOR='\033[0m'
RED = '\033[1;31m'

CHECK_IMAGETK = python3 -c "from tkinter import *" 

MODEL_FILE_NAME_BASE = 20180408-102900
MODEL_RELATIVE_DIR = ../../networks/facenet

FD_MODEL_FILE_NAME_BASE = face-detection-retail-0004
FD_MODEL_RELATIVE_DIR = ../../networks/face_detection_retail_0004

APP_RELATIVE_DIR = ../../apps/video_face_matcher

APP_NAME = video_face_matcher

CPU_EXT_PATH = ${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so

.PHONY: all
all: deps data


.PHONY: deps
deps: default_model
	@echo $(YELLOW)'\n'${APP_NAME}": Making dependencies..."$(NOCOLOR)


.PHONY: data
data: 
	@echo $(YELLOW)'\n'${APP_NAME}": Downloading required data for model..."$(NOCOLOR)
	mkdir -p validated_faces; \
	(cd validated_faces; mkdir -p Trump; cd Trump; wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics/trump.jpg)
	

.PHONY: compile_default_model
compile_default_model: 
	@echo $(YELLOW)'\n'${APP_NAME}": Compiling the model to IR..."$(NOCOLOR)
	@if [ -e ${MODEL_FILE_NAME_BASE}.bin ] && [ -e ${MODEL_FILE_NAME_BASE}.xml ]; \
	then \
		echo "- Facenet IR files found. No need to compile."; \
	else \
		echo "- Making Facenet IRs in Facenet project."; \
		(cd ${MODEL_RELATIVE_DIR}; make compile_model;); \
		echo "- Copying IRs to project folder..."; \
		(cd ${MODEL_RELATIVE_DIR}; cp ${MODEL_FILE_NAME_BASE}.bin ${APP_RELATIVE_DIR}; cp ${MODEL_FILE_NAME_BASE}.xml ${APP_RELATIVE_DIR};); \
	fi
	@if [ -e ${FD_MODEL_FILE_NAME_BASE}.bin ] && [ -e ${FD_MODEL_FILE_NAME_BASE}.xml ]; \
	then \
		echo "- Face detection IR files found. No need to compile."; \
	else \
		echo "- Making Face detection IRs in Facenet project."; \
		(cd ${FD_MODEL_RELATIVE_DIR}; make get_ir;); \
		echo "- Copying IRs to project folder..."; \
		(cd ${FD_MODEL_RELATIVE_DIR}; cp ${FD_MODEL_FILE_NAME_BASE}.bin ${APP_RELATIVE_DIR}; cp ${FD_MODEL_FILE_NAME_BASE}.xml ${APP_RELATIVE_DIR};); \
	fi
	
.PHONY: default_model
default_model: compile_default_model
	@echo $(YELLOW)'\n'${APP_NAME}": Making default models..."$(NOCOLOR)
	

.PHONY: run
run: run_py


.PHONY: run_py
run_py: data deps
	@echo $(YELLOW) '\n'${APP_NAME}": Running Python sample for Myriad..." $(NOCOLOR)
	@echo "Checking OpenVINO environment..."
	@if [ -z "$(INTEL_OPENVINO_DIR)" ] ; \
	then \
		echo "Please initiate the Intel OpenVINO environment by going to the installation directory for openvino and running the setupvars.sh file in the bin folder." ; \
		exit 1 ; \
	else \
		echo "Intel OpenVINO environment is already set!" ; \
	fi
	@echo "Checking tkinter..."
	@if $(shell) ${CHECK_IMAGETK} 2> /dev/null; \
	then \
		echo " - tkinter is already installed.\n"; \
	else \
		echo $(YELLOW)" - tkinter is not installed. Please run 'make install-reqs' to install all required packages. "$(NOCOLOR); \
		exit 1; \
	fi
	python3 ${APP_NAME}.py


.PHONY: run_cpu
run_cpu: data deps
	@echo $(YELLOW) '\n'${APP_NAME}": Running Python sample for CPU..." $(NOCOLOR)
	@echo "Checking OpenVINO environment..."
	@if [ -z "$(INTEL_OPENVINO_DIR)" ] ; \
	then \
		echo "Please initiate the Intel OpenVINO environment by going to the installation directory for openvino and running the setupvars.sh file in the bin folder." ; \
		exit 1 ; \
	else \
		echo "Intel OpenVINO environment is already set!" ; \
	fi
	@echo "Checking tkinter..."
	@if $(shell) ${CHECK_IMAGETK} 2> /dev/null; \
	then \
		echo " - tkinter is already installed.\n"; \
	else \
		echo $(YELLOW)" - tkinter is not installed. Please run 'make install-reqs' to install all required packages. "$(NOCOLOR); \
		exit 1; \
	fi
	python3 ${APP_NAME}.py -d=CPU -l=${CPU_EXT_PATH}
	

.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)'\n'${APP_NAME}": Checking application requirements...\n"$(NOCOLOR)
	@if $(shell) ${CHECK_IMAGETK} 2> /dev/null; \
	then \
		echo " - tkinter already installed.\n"; \
	else \
		echo $(YELLOW)"Installing tkinter... "$(NOCOLOR); \
		sudo apt-get install python3-tk; \
	fi; \


.PHONY: uninstall-reqs
uninstall-reqs: 
	@echo $(YELLOW)'\n'${APP_NAME}": Uninstalling requirements..."$(NOCOLOR)
	@echo $(YELLOW)"\n Checking tkinter..."$(NOCOLOR)
	@if $(shell) ${CHECK_IMAGETK} 2> /dev/null; \
	then \
		echo $(YELLOW)"\n - Uninstalling tkinter..."$(NOCOLOR); \
		sudo apt-get remove python3-tk; \
		sudo apt autoremove -y; \
	else \
		echo " - Uninstall cancelled. Requirement is not installed."; \
	fi; \
		

.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run or run_py "$(NOCOLOR)"- Runs Python example.";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- Shows this message.";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- Makes everything needed to run, but doesn't run.";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- Downloads required data.";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- Makes dependencies for project, prepares model etc.";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- Removes files created in this directory.\n";
	

.PHONY: clean
clean:
	@echo $(YELLOW)'\n'${APP_NAME}": Cleaning up files..."$(NOCOLOR);
	rm -f ${MODEL_FILE_NAME_BASE}.xml
	rm -f ${MODEL_FILE_NAME_BASE}.bin
	rm -f ${FD_MODEL_FILE_NAME_BASE}.xml
	rm -f ${FD_MODEL_FILE_NAME_BASE}.bin
	rm -rf validated_faces/Trump
	
	
