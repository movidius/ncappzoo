
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

# name of the graphfile to create.
GRAPH_FILENAME = graph


GET_PICTURES = wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics/licenses.txt; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics/elvis-presley-401920_640.jpg; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics/trump.jpg; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics/president-67550_640.jpg

.PHONY: all
all: prereqs pictures compile

.PHONY: pictures
pictures:
	@echo "\nmaking pictures"
	${GET_PICTURES};

.PHONY: prereqs
prereqs:
	@echo "\nmaking prereqs"
	@sed -i 's/\r//' *.py
	@chmod +x *.py	


.PHONY: compile
compile: googlenet tiny_yolo
	@echo "\nmaking compile"
	

.PHONY: run_py
run_py: prereqs compile
	@echo "\nmaking run_py"
	python3 ./facenet.py

.PHONY: opencv
opencv: 
	@echo "\nmaking opencv"
	./install-opencv-from_source.sh

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes everything needed to run but doesn't run";
	@echo "  make compile - compiles required networks with SDK compiler tool to create graph files";
	@echo "  make run_py - runs the street_cam_threaded.py python example program";
	@echo "  make clean - removes all created content"

.PHONY: clean
clean: 
	@echo "\nmaking clean"
	rm -f ${GRAPH_FILENAME}
	rm -f trump.jpg
	rm -f president-67550_640.jpg
	rm -f elvis-presley-401920_640.jpg
	rm -f licenses.txt


