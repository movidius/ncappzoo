
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

# filenames for the graph files that we'll copy to this directory.
SSD_MOBILENET_GRAPH_FILENAME = graph

GET_VIDEOS = wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/traffic_vid/licenses.txt; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/traffic_vid/bus_station_6094_960x540.mp4; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/traffic_vid/motorcycle_6098_shortened_960x540.mp4; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/traffic_vid/contrapicado_traffic_shortened_960x540.mp4; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/traffic_vid/police_car_6095_shortened_960x540.mp4; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/traffic_vid/scooters_5638_shortened_960x540.mp4

.PHONY: all
all: prereqs videos ssd_mobilenet

.PHONY: ssd_mobilenet
ssd_mobilenet: 
	@echo "\nmaking ssd_mobilenet"
	(cd ../../caffe/SSD_MobileNet; make compile; cd ../../apps/video_objects; cp ../../caffe/SSD_MobileNet/graph ./${SSD_MOBILENET_GRAPH_FILENAME};) 

.PHONY: videos
videos:
	@echo "\nmaking videos"
	${GET_VIDEOS};

.PHONY: prereqs
prereqs:
	@echo "\nmaking prereqs"
	@sed -i 's/\r//' *.py
	@chmod +x *.py

.PHONY: run_py
run_py: prereqs ssd_mobilenet videos
	@echo "\nmaking run_py"
	python3 ./video_objects.py


.PHONY: run
run: run_py

.PHONY: install-reqs
install-reqs: 
	@echo "\nmaking install-reqs"
	./install-opencv-from_source.sh

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes everything needed to run but doesn't run";
	@echo "  make videos - downloads example videos to run app with";
	@echo "  make ssd_mobilenet - makes and copies the compiled graph file";
	@echo "  make run - runs the python application";
	@echo "  make install-reqs - Installs requirements on your system."
	@echo "                      Will removes pip3 opencv and build from source and install a new version." ;
	@echo "                      Only needed once on your system and only if the its not already installed.";
	@echo "                      This may take a long time depending on your system.";
	@echo "  make clean - removes all created content"

.PHONY: clean
clean: 
	@echo "\nmaking clean"
	rm -f ${SSD_MOBILENET_GRAPH_FILENAME}
	rm -f *.mp4
	rm -f licenses.txt


