
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

# filenames for the graph files that we'll copy to this directory.
YOLO_TINY_GRAPH_FILENAME = yolo_tiny.graph
GOOGLENET_GRAPH_FILENAME = googlenet.graph
DOG_FILENAME = dog.jpg
IMAGE_DIR = ./images

GET_IMAGES = mkdir -p ${IMAGE_DIR}; \
             wget -c --no-cache -P ${IMAGE_DIR} https://www.nps.gov/sajh/learn/nature/images/Eagles-2_Rebecca.jpg; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/bird_images/master/jas_birds_10.jpg; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/bird_images/master/jas_birds_2.jpg; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/bird_images/master/jas_birds_8.jpg; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/bird_images/master/nps_birds_1.jpg; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/bird_images/master/nps_birds_2.jpg; \
             wget -c --no-cache -P ${IMAGE_DIR} https://raw.githubusercontent.com/nealvis/bird_images/master/nps_birds_3.jpg



.PHONY: all
all: prereqs ilsvrc12 compile images

.PHONY: images
images:
	@echo "\nmaking images"
	${GET_IMAGES};

.PHONY: ilsvrc12
ilsvrc12:
	@echo "\nmaking ilsvrc12"
	(cd ../../data/ilsvrc12; make)

.PHONY: prereqs
prereqs:
	@echo "\nmaking prereqs"
	@sed -i 's/\r//' birds.py
	@chmod +x birds.py	

.PHONY: googlenet
googlenet: 
	@echo "\nmaking googlenet"
	(cd ../../caffe/GoogLeNet; make compile; cd ../../apps/birds; cp ../../caffe/GoogLeNet/graph ./googlenet.graph;) 

.PHONY: tiny_yolo
tiny_yolo: 
	@echo "\nmaking tiny_yolo"
	(cd ../../caffe/TinyYolo; make compile; cd ../../apps/birds; cp ../../caffe/TinyYolo/graph ./yolo_tiny.graph;)


.PHONY: compile
compile: googlenet tiny_yolo
	@echo "\nmaking compile"
	

.PHONY: run_py
run_py: prereqs ilsvrc12 images compile
	@echo "\nmaking run_py"
	python3 ./birds.py

.PHONY: run
run: run_py
	@echo ""


.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes everything needed to run but doesn't run";
	@echo "  make compile - compiles required networks with SDK compiler tool to create graph files";
	@echo "  make run or run_py - runs the birds.py python example program";
	@echo "  make clean - removes all created content in this directory."

.PHONY: clean
clean: 
	@echo "\nmaking clean"
	rm -f ${YOLO_TINY_GRAPH_FILENAME}
	rm -f ${GOOGLENET_GRAPH_FILENAME}
	rm -rf ${IMAGE_DIR}


