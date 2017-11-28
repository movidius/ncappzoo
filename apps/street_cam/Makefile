
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

# filenames for the graph files that we'll copy to this directory.
YOLO_TINY_GRAPH_FILENAME = yolo_tiny.graph
GOOGLENET_GRAPH_FILENAME = googlenet.graph


GET_VIDEOS = wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/traffic_vid/licenses.txt; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/traffic_vid/bus_station_6094_960x540.mp4; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/traffic_vid/motorcycle_6098_shortened_960x540.mp4; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/traffic_vid/contrapicado_traffic_shortened_960x540.mp4; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/traffic_vid/police_car_6095_shortened_960x540.mp4; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/traffic_vid/scooters_5638_shortened_960x540.mp4

.PHONY: all
all: prereqs videos ilsvrc12 compile

.PHONY: videos
videos:
	@echo "\nmaking videos"
	${GET_VIDEOS};

.PHONY: ilsvrc12
ilsvrc12:
	@echo "\nmaking ilsvrc12"
	(cd ../../data/ilsvrc12; make)

.PHONY: prereqs
prereqs:
	@echo "\nmaking prereqs"
	@sed -i 's/\r//' street_cam.py
	@chmod +x street_cam.py	

.PHONY: googlenet
googlenet: 
	@echo "\nmaking googlenet"
	(cd ../../caffe/GoogLeNet; make compile; cd ../../apps/street_cam; cp ../../caffe/GoogLeNet/graph ./${GOOGLENET_GRAPH_FILENAME};) 

.PHONY: tiny_yolo
tiny_yolo: 
	@echo "\nmaking tiny_yolo"
	(cd ../../caffe/TinyYolo; make compile; cd ../../apps/street_cam; cp ../../caffe/TinyYolo/graph ./${YOLO_TINY_GRAPH_FILENAME};)


.PHONY: compile
compile: googlenet tiny_yolo
	@echo "\nmaking compile"
	

.PHONY: run_py
run_py: prereqs ilsvrc12 compile
	@echo "\nmaking run_py"
	python3 ./street_cam.py

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
	@echo "  make run_py - runs the street_cam.py python example program";
	@echo "  make opencv - removes pip3 opencv and builds from source then installs a new version" ;
	@echo "  make clean - removes all created content"

.PHONY: clean
clean: 
	@echo "\nmaking clean"
	rm -f ${YOLO_TINY_GRAPH_FILENAME}
	rm -f ${GOOGLENET_GRAPH_FILENAME}
	rm -f *.mp4
	rm -f licenses.txt


