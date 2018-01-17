
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

# name of the original model files from https://github.com/davidsandberg/facenet
MODEL_FILENAME_PREFIX_ORIG = model-20170512-110547.ckpt-250000
MODEL_FILENAME_DATA_ORIG = ${MODEL_FILENAME_PREFIX_ORIG}.data-00000-of-00001
MODEL_FILENAME_META_ORIG = ${MODEL_FILENAME_PREFIX_ORIG}.meta
MODEL_FILENAME_INDEX_ORIG = ${MODEL_FILENAME_PREFIX_ORIG}.index

MODEL_FILENAME_PREFIX_NCS = facenet_celeb_ncs
MODEL_DIR_NCS = ${MODEL_FILENAME_PREFIX_NCS}
GRAPH_FILENAME = ${MODEL_FILENAME_PREFIX_NCS}.graph
MODEL_FILENAME_DATA_NCS = ${MODEL_FILENAME_PREFIX_NCS}.data-00000-of-00001
MODEL_FILENAME_META_NCS = ${MODEL_FILENAME_PREFIX_NCS}.meta
MODEL_FILENAME_INDEX_NCS = ${MODEL_FILENAME_PREFIX_NCS}.index
MODEL_ZIP_FILENAME = 20170512-110547.zip
MODEL_UNZIP_DIR = 20170512-110547


GET_PICTURES = wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics/licenses.txt; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics/elvis-presley-401920_640.jpg; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics/trump.jpg; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics/president-67550_640.jpg


.PHONY: all
all: prereqs pictures compile

.PHONY: pictures
pictures:
	@echo "\nmaking pictures"
	MODEL_UNZIP_DIR${GET_PICTURES};

.PHONY: prereqs
prereqs:
	@echo "\nmaking prereqs"
	@sed -i 's/\r//' *.py
	@chmod +x *.py	


.PHONY: compile
compile: 
	( echo "\nmaking compile" ; \
	cd ${MODEL_DIR_NCS} ; \
	${NCCOMPILE} ${MODEL_FILENAME_META_NCS} -w ${MODEL_FILENAME_PREFIX_NCS} -s 12 -in input -on output -o ${GRAPH_FILENAME} ; \
	cp ${GRAPH_FILENAME} .. ; \	
	cd .. )

# mvNCCompile facenet_celeb_ncs.meta -w facenet_celeb_ncs -s 12 -in input -on output -o facenet_celeb_ncs.graph

.PHONY: model
model: 
	@echo "\nmaking model"
	@echo "Download " ${MODEL_ZIP_FILENAME} " from here: https://github.com/davidsandberg/facenet"
	unzip ${MODEL_ZIP_FILENAME}
	cd ${MODEL_UNZIP_DIR}
	mv ${MODEL_FILENAME_DATA_ORIG} ${MODEL_FILENAME_DATA_NCS} 
	mv ${MODEL_FILENAME_INDEX_ORIG} ${MODEL_FILENAME_INDEX_NCS} 
	mv ${MODEL_FILENAME_META_ORIG} ${MODEL_FILENAME_META_NCS} 
	cd ..


.PHONY: blahblah
blahblah:
	@if [ -e ${MODEL_FILENAME} ] ; \
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
	rm -f MODEL_FILENAME_INDEX_NCS
	rm -f MODEL_FILENAME_DATA_NCS
	rm -f MODEL_FILENAME_META_NCS


