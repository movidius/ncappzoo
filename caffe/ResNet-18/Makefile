
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck


TRAIN_PROTOTXT_FILENAME= train.prototxt
DEPLOY_PROTOTXT_FILENAME = deploy.prototxt
PROTOTXT_PATCH_FILENAME= resnet-18.prototxt.patch
GET_TRAIN_PROTOTXT = wget -P . https://raw.githubusercontent.com/HolmesShuan/ResNet-18-Caffemodel-on-ImageNet/3f035441361357a0f213269a5f63a6176ea28ebe/${TRAIN_PROTOTXT_FILENAME}

GRAPH_FILENAME = graph

CAFFEMODEL_FILENAME = resnet-18.caffemodel
CAFFEMODEL_URL = https://github.com/HolmesShuan/ResNet-18-Caffemodel-on-ImageNet

GET_CAFFEMODEL = wget --no-check-certificate 'https://onedrive.live.com/download?cid=7C725726AF404CFD&resid=7C725726AF404CFD%21110&authkey=%21ANF8hWlK9D8LGrk' -O resnet-18.caffemodel

.PHONY: all
all: profile check compile cpp 

.PHONY: prereqs
prereqs:
	@echo "\nmaking prereqs"
	(cd ../../data/ilsvrc12; make)
	@sed -i 's/\r//' run.py
	@chmod +x run.py

.PHONY: prototxt
prototxt: prereqs
	@echo "\nmaking prototxt"
	@if [ -e ${DEPLOY_PROTOTXT_FILENAME} ] ; \
	then \
		echo "Prototxt file already exists, skipping download."; \
	else \
		if ${GET_TRAIN_PROTOTXT} ; \
		then \
			cp ${TRAIN_PROTOTXT_FILENAME} ${DEPLOY_PROTOTXT_FILENAME} ; \
			echo "Training prototxt file downloaded, now patch for deployment" ; \
			patch ${DEPLOY_PROTOTXT_FILENAME} ${PROTOTXT_PATCH_FILENAME} ;\
		else \
			echo "Error, protoxt file failed to download."; \
			echo "Please check internet connection and retry."; \
		fi; \
	fi  

.PHONY: caffemodel
caffemodel: 
	@echo "\nmaking caffemodel"
	@if [ -e ${CAFFEMODEL_FILENAME} ] ; \
	then \
		echo "caffemodel file already exists, skipping download."; \
	else \
		if ${GET_CAFFEMODEL}; \
		then \
		    echo "Successfully downloaded caffemodel." ; \
		else \
			echo "\n***************" ; \
			echo "Error downloading caffemodel file: ${CAFFEMODEL_FILENAME}" ; \
			echo "Please check internet connection and retry or download manually at: "; \
			echo "\n${CAFFEMODEL_URL}\n"; \
			echo "***************" ; \
			rm -f ${CAFFEMODEL_FILENAME} ; \
			exit 1; \
		fi ; \
	fi  


.PHONY: profile
profile: prototxt
	@echo "\nmaking profile"
	${NCPROFILE} ${DEPLOY_PROTOTXT_FILENAME} -s 12

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
	${NCCOMPILE} -w ${CAFFEMODEL_FILENAME} -o ${GRAPH_FILENAME} -s 12 ${DEPLOY_PROTOTXT_FILENAME}

.PHONY: check
check: prototxt caffemodel
	@echo "\nmaking check"
	${NCCHECK} -w ${CAFFEMODEL_FILENAME} -i ../../data/images/nps_electric_guitar.png -s 12 -id 546 -S 255 -M ../../data/ilsvrc12/ilsvrc_2012_mean.npy ${DEPLOY_PROTOTXT_FILENAME}

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


clean: 
	@echo "\nmaking clean"
	rm -f ${CAFFEMODEL_FILENAME}
	rm -f ${GRAPH_FILENAME}
	rm -f ${DEPLOY_PROTOTXT_FILENAME}
	rm -f ${TRAIN_PROTOTXT_FILENAME}
	rm -f output.gv
	rm -f output.gv.svg
	rm -f output_report.html
	rm -f output_expected.npy
	rm -f zero_weights.caffemodel
	rm -f output_result.npy
	rm -f output_val.csv
	rm -f cpp/run_cpp
	
