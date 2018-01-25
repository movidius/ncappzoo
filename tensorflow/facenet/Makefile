
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

# name of the original model files from https://github.com/davidsandberg/facenet
MODEL_FILENAME_PREFIX_ORIG = model-20170512-110547.ckpt-250000
MODEL_FILENAME_DATA_ORIG = ${MODEL_FILENAME_PREFIX_ORIG}.data-00000-of-00001
MODEL_FILENAME_META_ORIG = model-20170512-110547.meta
MODEL_FILENAME_INDEX_ORIG = ${MODEL_FILENAME_PREFIX_ORIG}.index

MODEL_FILENAME_PREFIX_TEMP = facenet_celeb
MODEL_FILENAME_DATA_TEMP = ${MODEL_FILENAME_PREFIX_TEMP}.data-00000-of-00001
MODEL_FILENAME_META_TEMP = ${MODEL_FILENAME_PREFIX_TEMP}.meta
MODEL_FILENAME_INDEX_TEMP = ${MODEL_FILENAME_PREFIX_TEMP}.index


MODEL_ZIP_FILENAME = 20170512-110547.zip
MODEL_UNZIP_DIR = 20170512-110547
MODEL_FILENAME_PREFIX_NCS = facenet_celeb_ncs
MODEL_DIR_NCS = ${MODEL_UNZIP_DIR}/${MODEL_FILENAME_PREFIX_NCS}
GRAPH_FILENAME = ${MODEL_FILENAME_PREFIX_NCS}.graph
MODEL_FILENAME_DATA_NCS = ${MODEL_FILENAME_PREFIX_NCS}.data-00000-of-00001
MODEL_FILENAME_META_NCS = ${MODEL_FILENAME_PREFIX_NCS}.meta
MODEL_FILENAME_INDEX_NCS = ${MODEL_FILENAME_PREFIX_NCS}.index

COMPILE_FULL_COMMAND = ${NCCOMPILE} ${MODEL_FILENAME_META_NCS} -w ${MODEL_FILENAME_PREFIX_NCS} -s 12 -in input -on output -o ${GRAPH_FILENAME}


GET_PICTURES = wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics/licenses.txt; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics/elvis-presley-401920_640.jpg; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics/trump.jpg; \
             wget -c --no-cache -P . https://raw.githubusercontent.com/nealvis/media/master/face_pics/president-67550_640.jpg

INCEPTION_RESNET_PY = inception_resnet_v1.py
GET_INCEPTION_RESNET = wget -c --no-cache -P . https://raw.githubusercontent.com/davidsandberg/facenet/361c501c8b45183b9f4ad0171a9b1b20b2690d9f/src/models/inception_resnet_v1.py


# The facenet trained model is at this google drive url:
# https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk
# we will attempt to get it with by calling this script to do the wget command.
GET_FACENET_ZIPPED_MODEL = ./get_zipped_facenet_model.sh

.PHONY: all
all: prereqs pictures compile

.PHONY: pictures
pictures:
	@echo "\nmaking pictures"
	${GET_PICTURES};

.PHONY: prereqs
prereqs: pictures
	@echo "\nmaking prereqs" ; \
	if [ ! -e ${INCEPTION_RESNET_PY} ] ; \
	then \
		${GET_INCEPTION_RESNET} ; \
	else \
		echo "${INCEPTION_RESNET_PY} exists, not regetting it." ; \
	fi ; \
	sed -i 's/\r//' *.py ; \
	chmod +x *.py ; \


.PHONY: compile
compile: model
	@echo "\nmaking compile" ; \
	if [ -e ${GRAPH_FILENAME} ] ; \
	then \
		echo "graph file exists, skipping compilation." ; \
		echo "    if you want to re-compile, remove ${GRAPH_FILENAME}, and re-run" ; \
	else \
		cd ${MODEL_DIR_NCS} ; \
		echo "Command line: ${COMPILE_FULL_COMMAND}" ; \
		${COMPILE_FULL_COMMAND} ; \
		cp ${GRAPH_FILENAME} ../.. ; \
		cd ../.. ; \
	fi

.PHONY: zipped_model
zipped_model: 
	@echo "\nmaking zipped model" ; \
	if [ -e ${MODEL_ZIP_FILENAME} ] ; \
	then \
		echo "Zipped model already exists, skipping download" ; \
	else \
		${GET_FACENET_ZIPPED_MODEL} ; \
	fi

.PHONY: model
model: prereqs zipped_model
	@echo "\nmaking model" ; \
	if [ -e ${MODEL_ZIP_FILENAME} ] ; \
	then \
		echo "Zip file exists."; \
		if [ -d ${MODEL_UNZIP_DIR} ] ; \
		then \
			 echo "Zip file unzipped." ; \
		else \
			echo "Unzipping." ; \
			unzip ${MODEL_ZIP_FILENAME} ; \
		fi ; \
		cd ${MODEL_UNZIP_DIR} ; \
		if [ ! -e ${MODEL_FILENAME_DATA_TEMP} ] ; then mv ${MODEL_FILENAME_DATA_ORIG} ${MODEL_FILENAME_DATA_TEMP} ; \
		                                         else echo "data file exists" ; fi ; \
		if [ ! -e ${MODEL_FILENAME_INDEX_TEMP} ] ; then mv ${MODEL_FILENAME_INDEX_ORIG} ${MODEL_FILENAME_INDEX_TEMP} ; \
		                                         else echo "index file exists" ; fi ; \
		if [ ! -e ${MODEL_FILENAME_META_TEMP} ] ; then mv ${MODEL_FILENAME_META_ORIG} ${MODEL_FILENAME_META_TEMP} ; \
		                                         else echo "meta file exists" ; fi ; \
		if [ ! -e ${MODEL_FILENAME_PREFIX_NCS} ] ; \
		then \
			echo "Converted directory does not exist, doing conversion" ; \
			python3 ../convert_facenet.py model_base=${MODEL_FILENAME_PREFIX_TEMP}; \
		else \
			echo "Converted directory exists, skipping conversion. " ; \
			echo "    If you want to reconvert remove directory: ${MODEL_FILENAME_PREFIX_NCS}, and re-run" ; \
		fi ; \
		cd .. ; \
	else \
		echo "" ; \
		echo "***********************************************************************" ; \
		echo "Missing zipped model file.  Please download ${MODEL_ZIP_FILENAME}" ; \
		echo "from https://github.com/davidsandberg/facenet and put it in facenet base dir."; \
		echo "Then re-run the same make command." ; \
		echo "" ; \
		echo "Pro Tip: The direct google drive link to ${MODEL_ZIP_FILENAME} is:" ; \
		echo "https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk" ; \
		echo "***********************************************************************" ; \
		echo "" ; \
		exit 1 ; \
	fi

.PHONY: run_py
run_py: prereqs compile
	@echo "\nmaking run_py"
	python3 ./run.py

.PHONY: run
run: prereqs compile
	@echo "\nmaking run"
	python3 ./run.py

.PHONY: opencv
opencv: 
	@echo "\nmaking opencv"
	./install-opencv-from_source.sh

.PHONY: help
help:
	@echo "NCS FaceNet example requires that you download ${MODEL_ZIP_FILENAME}" ;
	@echo "from https://github.com/davidsandberg/facenet and put it in the facenet base directory" ;
	@echo "" ;
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes everything needed to run but doesn't run";
	@echo "  make compile - compiles required networks with SDK compiler tool to create graph files";
	@echo "  make model - process zip file and converts model for NCS"
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
	rm -rf ${MODEL_UNZIP_DIR}
	rm -f ${MODEL_ZIP_FILENAME}
	rm -f ${INCEPTION_RESNET_PY}


