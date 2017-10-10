
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:/opt/movidius/mvnc/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck


PROTOTXT_FILENAME= deploy.prototxt
GET_PROTOTXT = wget -P . https://github.com/DeepScale/SqueezeNet/raw/master/SqueezeNet_v1.0/${PROTOTXT_FILENAME}

CAFFEMODEL_FILENAME = squeezenet_v1.0.caffemodel
GET_CAFFEMODEL = wget -P . -N https://github.com/DeepScale/SqueezeNet/raw/master/SqueezeNet_v1.0/${CAFFEMODEL_FILENAME}


.PHONY: all
all: profile check compile cpp

.PHONY: prereqs
prereqs:
	@echo "\nmaking prereqs"
	(cd ../../data/ilsvrc12; make)
	@sed -i 's/\r//' run.py
	@chmod +x run.py
	@if [ -e LICENSE.squeezenet ] ; \
	then \
		echo "LICENSE file already exists"; \
	else \
		echo "Downloading LICENSE file"; \
		wget -P . -N https://github.com/DeepScale/SqueezeNet/raw/master/LICENSE; mv LICENSE LICENSE.squeezenet; \
	fi ; \


.PHONY: prototxt
prototxt: prereqs
	@echo "\nmaking prototxt"
	@if [ -e ${PROTOTXT_FILENAME} ] ; \
	then \
		echo "Prototxt file already exists"; \
	else \
		echo "Downloading Prototxt file"; \
		${GET_PROTOTXT}; \
		if [ -e ${PROTOTXT_FILENAME} ] ; \
		then \
			echo "modifying prototxt file."; \
			sed -i '11s/10/1/' ${PROTOTXT_FILENAME}; sed -i '530s/pad/#pad/' ${PROTOTXT_FILENAME}; \
		else \
			echo "***\nError - Could not download prototxt file. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi  

.PHONY: caffemodel
caffemodel: 
	@echo "\nmaking caffemodel"
	@if [ -e ${CAFFEMODEL_FILENAME} ] ; \
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

.PHONY: profile
profile: prototxt
	@echo "\nmaking profile"
	${NCPROFILE} ${PROTOTXT_FILENAME} -s 12

.PHONY: browse_profile
browse_profile: profile
	@echo "\nmaking browse_profile"
	@if [ -e output_report.html ] ; \
	then \
		firefox output_report.html & \
	else \
		@echo "***\nError - output_report.html not found" ; \
	fi ; 

.PHONY: compile
compile: prototxt caffemodel
	@echo "\nmaking compile"
	${NCCOMPILE} -w ${CAFFEMODEL_FILENAME} -s 12 ${PROTOTXT_FILENAME}

.PHONY: check
check: prototxt caffemodel
	@echo "\nmaking check"
#-${NCCHECK} -w ${CAFFEMODEL_FILENAME} -i ../../data/images/cat.jpg -s 12 -id 281 ${PROTOTXT_FILENAME} -S 255 -M ../../data/ilsvrc12/ilsvrc_2012_mean.npy
	${NCCHECK} -w ${CAFFEMODEL_FILENAME} -i ../../data/images/cat.jpg -s 12 -id 281  ${PROTOTXT_FILENAME} -S 255 -M 120

.PHONY: run
run: compile
	@echo "\nmaking run"
	./run.py

.PHONY: run_py
run_py: compile
	@echo "\nmaking run_py"
	./run.py

.PHONY:	cpp
cpp: cpp/run.cpp cpp/fp16.c
	@echo "\nmaking cpp"
	g++ cpp/run.cpp cpp/fp16.c -o cpp/run_cpp -lmvnc

.PHONY: run_cpp
run_cpp: compile cpp
	@echo "\nmaking run_cpp"
	cd cpp; ./run_cpp; cd ..

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: prototxt, profile, compile, check, cpp, run_py, run_cpp";
	@echo "  make prototxt - downloads and adds input shape to Caffe prototxt file";
	@echo "  make caffemodel - downloads the caffemodel for the network"
	@echo "  make compile - runs SDK compiler tool to compile the NCS graph file for the network";
	@echo "  make check - runs SDK checker tool to verify an NCS graph file";
	@echo "  make profile - runs the SDK profiler tool to profile the network creating output_report.html";
	@echo "  make browse_profile - runs the SDK profiler tool and brings up report in browser.";
	@echo "  make cpp - builds the run_cpp executable example";
	@echo "  make run_cpp - runs the run_cpp executable example program";
	@echo "  make run_py - runs the run.py python example program";
	@echo "  make clean - removes all created content"

clean_caffe_model:
	@echo "\nmaking clean_caffe_model"
	rm -f ${PROTOTXT_FILENAME}
	rm -f ${CAFFEMODEL_FILENAME}

clean: clean_caffe_model
	@echo "\nmaking clean"
	rm -f graph
	rm -f output.gv
	rm -f output.gv.svg
	rm -f output_report.html
	rm -f output_expected.npy
	rm -f zero_weights.caffemodel
	rm -f output_result.npy
	rm -f output_val.csv
	rm -f cpp/run_cpp
	rm -f LICENSE.squeezenet
