
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
compile: model
	@echo "\nmaking compile" ; \
	cd ${MODEL_DIR_NCS} ; \
	echo "Command line: ${COMPILE_FULL_COMMAND}" ; \
	${COMPILE_FULL_COMMAND} ; \
	cp ${GRAPH_FILENAME} ../.. ; \
	cd ../.. ;


.PHONY: old_comp
old_comp: model
	( echo "\nmaking compile" ; \
	cd ${MODEL_DIR_NCS} ; \
	${NCCOMPILE} ${MODEL_FILENAME_META_NCS} -w ${MODEL_FILENAME_PREFIX_NCS} -s 12 -in input -on output -o ${GRAPH_FILENAME} ; \
	cp ${GRAPH_FILENAME} ../.. ; \
	cd ../.. )

# mvNCCompile facenet_celeb_ncs.meta -w facenet_celeb_ncs -s 12 -in input -on output -o facenet_celeb_ncs.graph

.PHONY: model
model: 
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
			echo "    If you want to reconvert remove directory: ${MODEL_FILENAME_PREFIX_NCS}" ; \
		fi ; \
		cd .. ; \
	else \
		echo "" ; \
		echo "***********************************************************************" ; \
		echo "Missing zipped model file.  Please download ${MODEL_ZIP_FILENAME}" ; \
		echo "from https://github.com/davidsandberg/facenet"; \
		echo "Then re-run the same make command." ; \
		echo "***********************************************************************" ; \
		echo "" ; \
		exit 1 ; \
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


