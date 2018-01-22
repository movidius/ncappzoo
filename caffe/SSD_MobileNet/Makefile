
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

# filenames for the graph files that we'll copy to this directory.
SSD_MOBILENET_GRAPH_FILENAME = graph

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


.PHONY: all
all: prereqs prototxt caffemodel compile


.PHONY: prereqs
prereqs:
	@echo "\nmaking prereqs"
	@sed -i 's/\r//' *.py
	@chmod +x *.py

.PHONY: prototxt
prototxt: prereqs
	@echo "\nmaking prototxt"
	@if [ -e ${PATCHED_PROTOTXT_FILENAME} ] ; \
	then \
		echo "Prototxt file already exists"; \
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

.PHONY: old_caffemodel
old_caffemodel: 
	@echo "\nmaking caffemodel"
	@if [ -e ${CAFFEMODEL_FILENAME} ] ; \
	then \
		echo "caffemodel file already exists"; \
	else \
		echo "-----------------"; \
		echo "The caffe model file is at the google drive URL below."; \
		echo "For now the compiled graph file is included in this"; \
		echo "project directory so its not required to download the model."; \
		echo "or even compile it.  Its already there."; \
		echo "If you want to compile the model yourself please download it here:"; \
		echo ""; \
		echo "https://drive.google.com/open?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"; \
		echo "-----------------"; \
		echo "Attempting to download Caffe Model from google drive"; \
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


# If the model becomes available for download via wget do it like this:
#@echo "Downloading caffemodel file"; \
#${GET_CAFFEMODEL}; \
#if ! [ -e ${CAFFEMODEL_FILENAME} ] ; \
#then \
#	echo "***\nError - Could not download caffemodel file. Check network and proxy settings \n***\n"; \
#	exit 1; \
#fi ; \


.PHONY: compile
compile: caffemodel prototxt
	@echo "\nmaking compile"
	${NCCOMPILE} -w ${CAFFEMODEL_FILENAME} -s 12 ${PROTOTXT_FILENAME}

.PHONY: run_py
run_py: all
	@echo "\nmaking run_py"
	python3 ./run.py

.PHONY: run
run: run_py

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes everything needed to run but doesn't run";
	@echo "  make run_py - runs the street_cam.py python example program";
	@echo "  make clean - removes all created content"

.PHONY: clean
clean: 
	@echo "\nmaking clean"
	rm -f ./${PROTOTXT_FILENAME}
	rm -f ./${PATCHED_PROTOTXT_FILENAME}
	rm -f ${SSD_MOBILENET_GRAPH_FILENAME}
	rm -f ${CAFFEMODEL_FILENAME}


