
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

# filenames for the graph files that we'll copy to this directory.
SSD_MOBILENET_GRAPH_FILENAME = graph

# The caffe model file is here, but can't do wget for file on
# google drive.  Need to figure out how to download this from
# script or makefile.  For now the compiled graph file
# will be checked into the repository. The 'make compile' will be
# tell you to go download it manually.
#https://drive.google.com/open?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc

PROTOTXT_FILENAME= MobileNetSSD_deploy.prototxt
#PROTOTXT_FILENAME= mobile_net_ssd.prototxt

GET_PROTOTXT = wget -P . -O ${PROTOTXT_FILENAME} https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/${PROTOTXT_FILENAME}

CAFFEMODEL_FILENAME = MobileNetSSD_deploy.caffemodel
GET_CAFFEMODEL = wget -P . -N http://blah.com/${CAFFEMODEL_FILENAME}


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
	@if [ -e ${PROTOTXT_FILENAME} ] ; \
	then \
		echo "Prototxt file already exists"; \
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
compile: 
	@echo "\nmaking compile"
	@echo "-----------------"
	@echo "The compiled graph file for the network is included in the"
	@echo "project directory for now so there is no need to compile it."
	@echo "If you do want to compile it yourself you will need to first"
	@echo "download the model from the URL below and then run the command below."
	@echo ""
	@echo "URL for caffe model is:"
	@echo "https://drive.google.com/open?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"
	@echo ""
	@echo "Command to compile the model is: "
	@echo "${NCCOMPILE} -w ${CAFFEMODEL_FILENAME} -s 12 ${PROTOTXT_FILENAME}"
	@echo "-----------------"


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
	# rm -f ${SSD_MOBILENET_GRAPH_FILENAME}


