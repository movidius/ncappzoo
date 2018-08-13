
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

# filenames for the graph files that we'll copy to this directory.
SSD_MOBILENET_GRAPH_FILENAME = graph

# deploy.prototxt from internet before patching
PROTOTXT_FILENAME= mobilenet_ssd_1.0_deploy.prototxt 

# After patching deploy.prototxt the name is this
PATCHED_PROTOTXT_FILENAME= patched_${PROTOTXT_FILENAME}

# The name of the patch to apply to the downloaded prototxt file
PATCH_FOR_PROTOTXT_FILENAME = deploy_prototxt_update.patch

GET_PROTOTXT = wget -P . -O ${PROTOTXT_FILENAME} https://raw.githubusercontent.com/yuanyuanli85/MobileNet-SSD/ssd_voc_10class/voc_10class/${PROTOTXT_FILENAME}

CAFFEMODEL_FILENAME = MobileNetSSD_deploy.caffemodel

# The caffe model file is at this google drive link.
# https://drive.google.com/open?id=1C48ZyBwmimItmVzPBe_C_zGZDQTqAWxd
#
# This command does a wget for a google drive document ID 
GET_CAFFEMODEL = touch /tmp/cookies.txt; wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1C48ZyBwmimItmVzPBe_C_zGZDQTqAWxd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1C48ZyBwmimItmVzPBe_C_zGZDQTqAWxd" -O ${CAFFEMODEL_FILENAME} && rm -rf /tmp/cookies.txt


.PHONY: all
all: prereqs prototxt caffemodel compile


.PHONY: data
data: 
	@echo "\ndownload data and create lmdb"	
	@(cd ./data && ./download_data.sh && cd ../)
	@(./create_lmdb.sh)

.PHONY: train
train: 
	@echo "\nstart training"
	@(./train.sh)

.PHONY: prereqs
prereqs:
	@echo "\nmaking prereqs"
	@sed -i 's/\r//' *.py
	@chmod +x *.py

.PHONY: prototxt
prototxt: 
	@echo "\nmaking prototxt"
	@if [ -e ${PROTOTXT_FILENAME} ] ; \
	then \
		echo "Prototxt file already exists, skipping download."; \
	else \
		echo "Downloading Prototxt file"; \
		${GET_PROTOTXT}; \
		if [ -e ${PROTOTXT_FILENAME} ] ; \
		then \
			echo "prototxt file downloaded."; \
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
                echo "https://drive.google.com/open?id=1C48ZyBwmimItmVzPBe_C_zGZDQTqAWxd"; \
		echo ""; \
		${GET_CAFFEMODEL}; \
		if ! [ -e ${CAFFEMODEL_FILENAME} ] ; \
		then \
			echo "caffemodel download failed from url: "; \
			echo "https://drive.google.com/open?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"; \
			echo "Please download it manually or check internet connection and retry."; \
		fi; \
	fi


.PHONY: compile
compile: caffemodel prototxt
	@echo "\nmaking compile"; \
	if [ -e ${SSD_MOBILENET_GRAPH_FILENAME} ] ; \
	then \
		echo "NCS graph file already exists, skipping compile."; \
	else \
		${NCCOMPILE} -w ${CAFFEMODEL_FILENAME} -o ${SSD_MOBILENET_GRAPH_FILENAME} -s 12 ${PROTOTXT_FILENAME}; \
	fi

.PHONY: run_py
run_py: all
	@echo "\nmaking run_py"
	python3 ./run.py

.PHONY: run
run: run_py

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help  - shows this message";
	@echo "  make data  - downloads data, converts the annotations and generate the lmdb";
	@echo "  make train - start to train the network";
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


