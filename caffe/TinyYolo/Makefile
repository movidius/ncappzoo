
NCCOMPILE = mo.py

GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'

MODEL_BASE_NAME = tiny-yolo-v1_53000

PROTOTXT_FILENAME= tiny-yolo-v1.prototxt
GET_PROTOTXT = wget --no-cache -P . http://ncs-forum-uploads.s3.amazonaws.com/ncappzoo/tiny_yolo/${PROTOTXT_FILENAME}

CAFFEMODEL_FILENAME = tiny-yolo-v1_53000.caffemodel
GET_CAFFEMODEL = wget --no-cache -P . -N http://ncs-forum-uploads.s3.amazonaws.com/ncappzoo/tiny_yolo/${CAFFEMODEL_FILENAME}


.PHONY: all
all: data deps


.PHONY: deps
deps: prototxt caffemodel
	@echo $(YELLOW)"\nTiny yolo v1: Making dependencies..."$(NOCOLOR)

	
.PHONY: data
data:
	@echo $(YELLOW)"\nTiny yolo v1: Downloading required data... No data needed."$(NOCOLOR)


.PHONY: prototxt
prototxt: 
	@echo $(YELLOW)"\nTiny yolo v1: making prototxt"$(NOCOLOR)
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
	@echo $(YELLOW)"\nTiny yolo v1: making caffemodel"$(NOCOLOR)
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


.PHONY: compile_model
compile_model: prototxt caffemodel
	@echo $(YELLOW)"\nTiny yolo v1: compiling model to IR..."$(NOCOLOR)
	@if [ -e ${MODEL_BASE_NAME}.xml ] && [ -e ${MODEL_BASE_NAME}.bin ] ; \
	then \
		echo "no need to compile." ; \
	else \
	    ${NCCOMPILE} --input_model=${CAFFEMODEL_FILENAME} --input_proto=${PROTOTXT_FILENAME} --data_type=FP16 --scale_values=[255] --reverse_input_channels; \
	fi
	

.PHONY: run_py
run_py: compile_model
	@echo $(YELLOW)"\nTiny yolo v1:making run_py"$(NOCOLOR)
	python3 ./run.py


.PHONY: run
run: compile_model
	@echo $(YELLOW)"\nTiny yolo v1:making run"$(NOCOLOR)
	python3 ./run.py


.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo "  make help "$(NOCOLOR)"- shows this message";
	@echo "  make run "$(NOCOLOR)"- runs the run.py python example program";
	@echo "  make all "$(NOCOLOR)"- makes the following: prototxt, profile, compile, check, cpp, run_py, run_cpp";
	@echo "  make prototxt "$(NOCOLOR)"- downloads and adds input shape to Caffe prototxt file";
	@echo "  make caffemodel "$(NOCOLOR)"- downloads the caffemodel for the network"
	@echo "  make compile "$(NOCOLOR)"- runs SDK compiler tool to compile the NCS graph file for the network";
	@echo "  make clean "$(NOCOLOR)"- removes all created content"

clean_caffe_model:
	@echo $(YELLOW)"\nTiny yolo v1: making clean_caffe_model..."$(NOCOLOR)
	rm -f ${PROTOTXT_FILENAME}
	rm -f ${CAFFEMODEL_FILENAME}

clean: clean_caffe_model
	@echo $(YELLOW)"\nTiny yolo v1: making clean..."$(NOCOLOR)
	rm -f ${GRAPH_FILENAME}
	rm -f *.xml
	rm -f *.bin
	rm -f *.mapping
	

