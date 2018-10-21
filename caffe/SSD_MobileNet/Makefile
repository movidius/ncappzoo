
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

# filenames for the graph files that we'll copy to this directory.
SSD_MOBILENET_GRAPH_FILENAME = graph

# deploy.prototxt from internet before patching
PROTOTXT_FILENAME= deploy.prototxt

# After patching deploy.prototxt the name is this
PATCHED_PROTOTXT_FILENAME= patched_${PROTOTXT_FILENAME}

# The name of the patch to apply to the downloaded prototxt file
PATCH_FOR_PROTOTXT_FILENAME = deploy_prototxt_update.patch

GET_PROTOTXT = wget -P . -O ${PROTOTXT_FILENAME} https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/${PROTOTXT_FILENAME}

# The caffe weights file name
CAFFEMODEL_FILENAME = mobilenet_iter_73000.caffemodel

# This command does a wget for the caffe weights file from github
GET_CAFFEMODEL = wget -P . https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/${CAFFEMODEL_FILENAME}

# The merge batchnorm python script. This script generates a non-batchnorm version of the model
MERGE_BN_FILENAME = merge_bn.py
# This command gets the merge_bn script
GET_MERGE_BN = wget -P . https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/${MERGE_BN_FILENAME}

# names of the output files from merge_bn python script
NO_BN_CAFFEMODEL_FILENAME = no_bn.caffemodel
NO_BN_PROTOTXT_FILENAME = no_bn.prototxt

.PHONY: all
all: prereqs prototxt caffemodel compile


.PHONY: prereqs
prereqs:
	@echo "\nmaking prereqs"
	@sed -i 's/\r//' *.py
	@chmod +x *.py

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
		echo "https://github.com/chuanqi305/MobileNet-SSD/blob/master/mobilenet_iter_73000.caffemodel"; \
		echo ""; \
		${GET_CAFFEMODEL}; \
		if ! [ -e ${CAFFEMODEL_FILENAME} ] ; \
		then \
			echo "caffemodel download failed from url: "; \
			echo "https://github.com/chuanqi305/MobileNet-SSD/blob/master/mobilenet_iter_73000.caffemodel"; \
			echo "Please download it manually or check internet connection and retry."; \
		fi; \
	fi; \

.PHONY: merge_bn
merge_bn: 
	if [ -e ${MERGE_BN_FILENAME} ] ; \
	then \
		echo "merge_bn.py already exists. skipping download."; \
	else \
		echo ""; \
		echo "Attempting download of merge_bn.py from this url: "; \
		echo "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/merge_bn.py"; \
		${GET_MERGE_BN}; \
		python3 merge_bn.py --model ${PROTOTXT_FILENAME} --weight ${CAFFEMODEL_FILENAME}; \
	fi; \


.PHONY: compile
compile: caffemodel prototxt merge_bn
	@echo "\nmaking compile"; \
	if [ -e ${SSD_MOBILENET_GRAPH_FILENAME} ] ; \
	then \
		echo "NCS graph file already exists, skipping compile."; \
	else \
		${NCCOMPILE} -w ${NO_BN_CAFFEMODEL_FILENAME} -o ${SSD_MOBILENET_GRAPH_FILENAME} -s 12 ${NO_BN_PROTOTXT_FILENAME}; \
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
	@echo "  make help - shows this message";
	@echo "  make all - makes everything needed to run but doesn't run.";
	@echo "  make prototxt - downloads prototxt file.";
	@echo "  make caffemodel - downloads trained model.";
	@echo "  make compile - compiles the trained model.";
	@echo "  make run - runs the python example program.";
	@echo "  make clean - removes all created content."

.PHONY: clean
clean: 
	@echo "\nmaking clean"
	rm -f ./${PROTOTXT_FILENAME}
	rm -f ./${PATCHED_PROTOTXT_FILENAME}
	rm -f ${SSD_MOBILENET_GRAPH_FILENAME}
	rm -f ${CAFFEMODEL_FILENAME}
	rm -f ${MERGE_BN_FILENAME}
	rm -f ${NO_BN_CAFFEMODEL_FILENAME}
	rm -f ${NO_BN_PROTOTXT_FILENAME}
	rm -f output_expected.npy
	rm -f Fathom_expected.npy



