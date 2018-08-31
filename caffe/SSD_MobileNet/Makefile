#NCCOMPILE = python /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo.py
NCCOMPILE = mo.py


# deploy.prototxt from internet before patching
PROTOTXT_FILENAME= MobileNetSSD_deploy.prototxt

# After patching deploy.prototxt the name is this
PATCHED_PROTOTXT_FILENAME= patched_${PROTOTXT_FILENAME}

# The name of the patch to apply to the downloaded prototxt file
PATCH_FOR_PROTOTXT_FILENAME = deploy_prototxt_update.patch

GET_PROTOTXT = wget -P . -O ${PROTOTXT_FILENAME} https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/${PROTOTXT_FILENAME}

CAFFEMODEL_FILENAME = MobileNetSSD_deploy.caffemodel

# The caffe model file is at this google drive link.
# https://drive.google.com/open?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc
#
# This command does a wget for a google drive document ID 
GET_CAFFEMODEL = touch /tmp/cookies.txt; wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc" -O ${CAFFEMODEL_FILENAME} && rm -rf /tmp/cookies.txt


GRAPH_FILENAME_BASE =  MobileNetSSD_deploy

.PHONY: all
all: deps data compile_model compile_cpp


.PHONY: prototxt
prototxt: 
	@echo "\nmaking prototxt"
	@if [ -e ${PATCHED_PROTOTXT_FILENAME} ] ; \
	then \
		echo "Prototxt file already exists, skipping download."; \
	else \
		echo "Downloading Prototxt file"; \
		${GET_PROTOTXT}; \
		if [ -e ${PROTOTXT_FILENAME} ] ; \
		then \
			echo "prototxt file downloaded."; \
			echo "patching prototxt."; \
			patch ${PROTOTXT_FILENAME}  -i ${PATCH_FOR_PROTOTXT_FILENAME} -o ${PATCHED_PROTOTXT_FILENAME}; \
		else \
			echo "***\nError - Could not download prototxt file. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi  


.PHONY: caffemodel
caffemodel: 
	@echo "\nmaking caffemodel"; \
	if [ -e ${CAFFEMODEL_FILENAME} ] ; \
	then \
		echo "caffemodel already exists, skipping download."; \
	else \
		echo ""; \
		echo "Attempting download of caffemodel file from this url: "; \
		echo "https://drive.google.com/open?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"; \
		echo ""; \
		${GET_CAFFEMODEL}; \
		if ! [ -e ${CAFFEMODEL_FILENAME} ] ; \
		then \
			echo "caffemodel download failed from url: "; \
			echo "https://drive.google.com/open?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"; \
			echo "Please download it manually or check internet connection and retry."; \
		fi; \
	fi

.PHONY: profile
profile: 
	@echo "\nProfile not implemented yet"

.PHONY: compile_model
compile_model: deps
	@echo "\nmaking compile"; \
	if [ -e ${GRAPH_FILENAME_BASE}.xml ] ; \
	then \
		echo "compiled model file already exists, skipping compile."; \
	else \
		${NCCOMPILE} --data_type FP16 --input_model ${CAFFEMODEL_FILENAME} --input_proto ${PROTOTXT_FILENAME};\
	fi

.PHONY: deps
deps: caffemodel prototxt
	@echo "\nmaking deps"

.PHONY: data
data: 
	@echo "\nmaking data - no data required for this project"

.PHONY: check
check: 
	@echo "\nCheck not implemented yet."


.PHONY: compile_cpp
compile_cpp: 
	@echo "\ncompile_cpp not implemented yet."

.PHONY: run
run: deps compile_model compile_cpp
	@echo "\nrun not implemented yet."

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - make all - everything needed to run, but doesn't run";
	@echo "  make data - downloads required data.";
	@echo "  make deps - makes dependencies for project, prepares model etc.";
	@echo "  make prototxt - downloads and adds input shape to Caffe prototxt file";
	@echo "  make caffemodel - downloads the caffemodel for the network"
	@echo "  make compile_model - runs model compiler for the network";
	@echo "  make compile_cpp - builds the run_cpp executable example";
	@echo "  make run - builds and runs the C++ example program";
	@echo "  make clean - removes all created content"

.PHONY: clean
clean: 
	@echo "\nmaking clean"
	rm -f ./${PROTOTXT_FILENAME}
	rm -f ./${PATCHED_PROTOTXT_FILENAME}
	rm -f ${CAFFEMODEL_FILENAME}
	rm -f ${GRAPH_FILENAME_BASE}.bin
	rm -f ${GRAPH_FILENAME_BASE}.mapping.json
	rm -f ${GRAPH_FILENAME_BASE}.xml



