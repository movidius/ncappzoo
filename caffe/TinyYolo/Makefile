
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:/opt/movidius/mvnc/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

GRAPH_FILENAME = graph
GET_GRAPH = wget --no-cache -P . -o ${GRAPH_FILENAME} http://ncs-forum-uploads.s3.amazonaws.com/ncappzoo/tiny_yolo/yolo_tiny.graph

PROTOTXT_FILENAME= tiny-yolo-v1.prototxt
GET_PROTOTXT = wget --no-cache -P . http://ncs-forum-uploads.s3.amazonaws.com/ncappzoo/tiny_yolo/${PROTOTXT_FILENAME}

CAFFEMODEL_FILENAME = tiny-yolo-v1_53000.caffemodel
GET_CAFFEMODEL = wget --no-cache -P . -N http://ncs-forum-uploads.s3.amazonaws.com/ncappzoo/tiny_yolo/${CAFFEMODEL_FILENAME}


.PHONY: all
all: profile compile 

.PHONY: prototxt
prototxt: 
	@echo "\nmaking prototxt"
	@if [ -e ${PROTOTXT_FILENAME} ] ; \
	then \
		echo "Prototxt file already exists"; \
	else \
		echo "Downloading Prototxt file"; \
		${GET_PROTOTXT}; \
		if [ -e ${PROTOTXT_FILENAME} ] ; \
		then \
			echo "got prototext file." ; \
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
		echo "Downloading caffemodel file"; \
		${GET_CAFFEMODEL}; \
		if ! [ -e ${CAFFEMODEL_FILENAME} ] ; \
		then \
			echo "***\nError - Could not download caffemodel file. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi  

.PHONY: profile
profile: prototxt
	@echo "\nmaking profile"
	${NCPROFILE} ${PROTOTXT_FILENAME} -s 12

.PHONY: browse_profile
browse_profile: profile
	@echo "\nmaking browse_profile"
	@if [ -e output_report.html ] ; \
	then \
		firefox output_report.html & \
	else \
		@echo "***\nError - output_report.html not found" ; \
	fi ; 

.PHONY: compile
compile: prototxt caffemodel
	@echo "\nmaking compile"
	${NCCOMPILE} -o ${GRAPH_FILENAME} -w ${CAFFEMODEL_FILENAME} -s 12 ${PROTOTXT_FILENAME}

.PHONY: graph
graph: 
	@echo "\nmaking (downloading) graph"
	@if [ -e ${GRAPH_FILENAME} ] ; \
	then \
		echo "graph file already exists"; \
	else \
		${GET_GRAPH}; \
		if ! [ -e ${GRAPH_FILENAME} ] ; \
		then \
			echo "***\nError - Could not download graph file. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi ;

.PHONY: run_py
run_py: compile
	@echo "\nmaking run_py"
	python3 ./run.py

.PHONY: run
run: compile
	@echo "\nmaking run"
	python3 ./run.py

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: prototxt, profile, compile, check, cpp, run_py, run_cpp";
	@echo "  make prototxt - downloads and adds input shape to Caffe prototxt file";
	@echo "  make caffemodel - downloads the caffemodel for the network"
	@echo "  make compile - runs SDK compiler tool to compile the NCS graph file for the network";
	@echo "  make profile - runs the SDK profiler tool to profile the network creating output_report.html";
	@echo "  make browse_profile - runs the SDK profiler tool and brings up report in browser.";
	@echo "  make run - runs the run.py python example program";
	@echo "  make clean - removes all created content"

clean_caffe_model:
	@echo "\nmaking clean_caffe_model"
	rm -f ${PROTOTXT_FILENAME}
	rm -f ${CAFFEMODEL_FILENAME}

clean: clean_caffe_model
	@echo "\nmaking clean"
	rm -f ${GRAPH_FILENAME}
	rm -f output.gv
	rm -f output.gv.svg
	rm -f output_report.html
	rm -f output_expected.npy
	rm -f zero_weights.caffemodel
	rm -f output_result.npy
	rm -f output_val.csv

