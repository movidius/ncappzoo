
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

IR = tiny-yolo-v1_53000.xml
IMG = ../../data/images/nps_chair.png

.PHONY: all
all: data deps compile_model


.PHONY: deps
deps: prototxt caffemodel
	@echo $(YELLOW)"\nTiny Yolo v1: Making dependencies..."$(NOCOLOR)

	
.PHONY: data
data:
	@echo $(YELLOW)"\nTiny Yolo v1: Downloading required data... No data needed."$(NOCOLOR)


.PHONY: prototxt
prototxt: 
	@echo $(YELLOW)"\nTiny Yolo v1: Downloading prototxt file..."$(NOCOLOR)
	@if [ -e ${PROTOTXT_FILENAME} ] ; \
	then \
		echo " - Prototxt file already exists"; \
	else \
		echo "Downloading Prototxt file"; \
		${GET_PROTOTXT}; \
		if [ -e ${PROTOTXT_FILENAME} ] ; \
		then \
			echo " - Got prototext file." ; \
		else \
			echo "***\nError - Could not download prototxt file. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi 

.PHONY: caffemodel
caffemodel: 
	@echo $(YELLOW)"\nTiny Yolo v1: Downloading caffemodel file..."$(NOCOLOR)
	@if [ -e ${CAFFEMODEL_FILENAME} ] ; \
	then \
		echo " - Caffemodel file already exists"; \
	else \
		echo " - Downloading caffemodel file"; \
		${GET_CAFFEMODEL}; \
		if ! [ -e ${CAFFEMODEL_FILENAME} ] ; \
		then \
			echo "***\nError - Could not download caffemodel file. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi  


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\nTiny Yolo v1: Checking application requirements...\n"$(NOCOLOR)
	@echo "No requirements needed."
	
	
.PHONY: compile_model
compile_model: prototxt caffemodel
	@echo $(YELLOW)"\nTiny Yolo v1: Compiling model to IR..."$(NOCOLOR)
	@if [ -e ${MODEL_BASE_NAME}.xml ] && [ -e ${MODEL_BASE_NAME}.bin ] ; \
	then \
		echo " - IR files exist. No need to compile." ; \
	else \
	    ${NCCOMPILE} --input_model=${CAFFEMODEL_FILENAME} --input_proto=${PROTOTXT_FILENAME} --data_type=FP16 --scale_values=[255] --reverse_input_channels || (echo $(RED)"Make sure your paths are correct and the OpenVINO environment variables are set using the "$(YELLOW)"setupvars.sh"$(RED)" script found in <your OpenVINO install location>/bin/ folder."$(NOCOLOR);  exit 1;); \
	fi
	

.PHONY: run_py
run_py: install-reqs data deps compile_model
	@echo $(YELLOW)"\nTiny Yolo v1: Running the python sample..."$(NOCOLOR)
	python3 ./tiny_yolo_v1.py --image=${IMG} --ir=${IR}


.PHONY: run
run: run_py
	@echo $(YELLOW)"\nTiny Yolo v1: Running the python sample..."$(NOCOLOR)


.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make run "$(NOCOLOR)"- runs the run.py python example program";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes the following: data deps compile_model";
	@echo $(YELLOW)"  make prototxt "$(NOCOLOR)"- downloads and adds input shape to Caffe prototxt file";
	@echo $(YELLOW)"  make caffemodel "$(NOCOLOR)"- downloads the caffemodel for the network"
	@echo $(YELLOW)"  make compile "$(NOCOLOR)"- runs Model Optimizer compiler tool to compile an IR file for the network";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content"


clean_caffe_model:
	@echo $(YELLOW)"\nTiny Yolo v1: making clean_caffe_model..."$(NOCOLOR)
	rm -f ${PROTOTXT_FILENAME}
	rm -f ${CAFFEMODEL_FILENAME}


clean: clean_caffe_model
	@echo $(YELLOW)"\nTiny Yolo v1: making clean..."$(NOCOLOR)
	rm -f ${MODEL_BASE_NAME}.xml
	rm -f ${MODEL_BASE_NAME}.bin
	rm -f ${MODEL_BASE_NAME}.mapping
	

