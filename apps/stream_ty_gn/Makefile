
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

# filenames for the graph files that we'll copy to this directory.
YOLO_TINY_GRAPH_FILENAME = yolo_tiny.graph
GOOGLENET_GRAPH_FILENAME = googlenet.graph


.PHONY: all
all: prereqs ilsvrc12 compile

.PHONY: ilsvrc12
ilsvrc12:
	@echo "\nmaking ilsvrc12"
	(cd ../../data/ilsvrc12; make)

.PHONY: prereqs
prereqs:
	@echo "\nmaking prereqs"
	@sed -i 's/\r//' stream_ty_gn.py
	@chmod +x stream_ty_gn.py	

.PHONY: googlenet
googlenet: 
	@echo "\nmaking googlenet"
	(cd ../../caffe/GoogLeNet; make compile; cd ../../apps/stream_ty_gn; cp ../../caffe/GoogLeNet/graph ./${GOOGLENET_GRAPH_FILENAME};) 

.PHONY: tiny_yolo
tiny_yolo: 
	@echo "\nmaking tiny_yolo"
	(cd ../../caffe/TinyYolo; make compile; cd ../../apps/stream_ty_gn; cp ../../caffe/TinyYolo/graph ./${YOLO_TINY_GRAPH_FILENAME};)


.PHONY: compile
compile: googlenet tiny_yolo
	@echo "\nmaking compile"
	

.PHONY: run_py
run_py: prereqs ilsvrc12 compile
	@echo "\nmaking run_py"
	python3 ./stream_ty_gn.py

.PHONY: run
run: prereqs ilsvrc12 compile
	@echo "\nmaking run"
	python3 ./stream_ty_gn.py

.PHONY: install-reqs
install-reqs: 
	@echo "\nmaking install-reqs"
	./install-opencv-from_source.sh

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes everything needed to run but doesn't run";
	@echo "  make compile - compiles required networks with SDK compiler tool to create graph files";
	@echo "  make run - runs the program";
	@echo "  make install-reqs - Installs requirements on your system."
	@echo "                      Will removes pip3 opencv and build from source and install a new version" ;
	@echo "                      Only needed once on your system and only if the its not already installed.";
	@echo "                      This may take a long time depending on your system.";
	@echo "  make clean - removes all created content"

.PHONY: clean
clean: 
	@echo "\nmaking clean"
	rm -f ${YOLO_TINY_GRAPH_FILENAME}
	rm -f ${GOOGLENET_GRAPH_FILENAME}


