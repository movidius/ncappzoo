

GOOGLENET_RELATIVE_DIR = ../../caffe/GoogLeNet
GOOGLENET_GRAPH_FILENAME_BASE = bvlc_googlenet

IMAGES_RELATIVE_DIR = ../../data/images
TEST_IMAGE_FILENAME = nps_electric_guitar.png

.PHONY: all
all: deps data compile_cpp

.PHONY: googlenet_data
googlenet_data: 
	@echo "\nmaking googlenet_data"
	(cd ${GOOGLENET_RELATIVE_DIR}; make data; cd -) 

.PHONY: googlenet_deps
googlenet_deps: 
	@echo "\nmaking googlenet_deps"
	(cd ${GOOGLENET_RELATIVE_DIR}; make deps; cd -) 

.PHONY: googlenet_compile_model
googlenet_compile_model: 
	@echo "\nmaking googlenet_compile_model"
	(cd ${GOOGLENET_RELATIVE_DIR}; make compile_model; cd -) 

.PHONY: data
data: googlenet_data
	@echo "\nmaking data"

.PHONY: deps
deps: googlenet_deps googlenet_compile_model
	@echo "\nmaking deps"

.PHONY: run
run: run_cpp
	@echo "\nmaking run"

.PHONY:	compile_cpp
compile_cpp:  
	@echo "\nmaking compile_cpp"
	(mkdir -p build; cd build; cmake ..; make)

.PHONY: run_cpp
run_cpp: deps data compile_cpp
	@echo "\nmaking run_cpp"
	./build/simple_classifier ${GOOGLENET_RELATIVE_DIR}/${GOOGLENET_GRAPH_FILENAME_BASE}.xml ${IMAGES_RELATIVE_DIR}/${TEST_IMAGE_FILENAME}; cd ..

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - everything needed to run, but doesn't run";
	@echo "  make data - downloads required data.";
	@echo "  make deps - makes dependencies to run project, prepares a model etc.";
	@echo "  make compile_cpp - builds the run_cpp executable example";
	@echo "  make run_cpp - builds and runs the C++ example program";
	@echo "  make clean - removes all created content"

clean: 
	@echo "\nmaking clean"
	rm -rf build
