
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

# filenames for the graph file that we'll copy to this directory.
GRAPH_FILENAME = facenet_celeb_ncs.graph

.PHONY: all
all: prereqs facenet

.PHONY: prereqs
prereqs:
	@echo "\nmaking prereqs"
	@sed -i 's/\r//' *.py
	@chmod +x *.py	

.PHONY: facenet
facenet: 
	@echo "\nmaking facenet"
	(cd ../../tensorflow/facenet; make compile; cd ../../apps/video_face_matcher; cp ../../tensorflow/facenet/${GRAPH_FILENAME} ./${GRAPH_FILENAME};) 


.PHONY: compile
compile: facenet
	@echo "\nmaking compile"
	

.PHONY: run_py
run_py: prereqs facenet
	@echo "\nmaking run_py"
	python3 ./video_face_matcher.py

.PHONY: run
run: prereqs facenet
	@echo "\nmaking run"
	python3 ./video_face_matcher.py

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
	@echo "  make run - runs the street_cam_threaded.py python program";
	@echo "  make install-reqs - Installs requirements on your system."
	@echo "                      Will removes pip3 opencv and build from source and install a new version." ;
	@echo "                      Only needed once on your system and only if the its not already installed.";
	@echo "                      This may take a long time depending on your system.";
	@echo "  make clean - removes all created content"

.PHONY: clean
clean: 
	@echo "\nmaking clean"
	rm -f ${GRAPH_FILENAME}


