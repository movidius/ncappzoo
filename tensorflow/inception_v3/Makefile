
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:/opt/movidius/mvnc/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

.IGNORE: profile check compile

WEIGHTS_FILENAME = inception_v3_2016_08_28.tar.gz
GET_WEIGHTS = (wget http://download.tensorflow.org/models/${WEIGHTS_FILENAME} && tar zxf ${WEIGHTS_FILENAME} && rm ${WEIGHTS_FILENAME})

MODEL_FILENAME = output/inception-v3.meta
CONV_SCRIPT = ./inception-v3.py

INPUT_NODE_FLAG = -in=input
OUTPUT_NODE_FLAG = -on=InceptionV3/Predictions/Reshape_1

.PHONY: all
all: profile check compile

.PHONY: prereqs
prereqs:
	(cd ../../data/ilsvrc12; make)
	@sed -i 's/\r//' run.py
	@chmod +x run.py

.PHONY: profile
profile: weights
	${NCPROFILE} -s 12 ${MODEL_FILENAME} ${INPUT_NODE_FLAG} ${OUTPUT_NODE_FLAG}

.PHONY: browse_profile
browse_profile: weights
	${NCPROFILE} -s 12 ${MODEL_FILENAME} ${INPUT_NODE_FLAG} ${OUTPUT_NODE_FLAG}
	firefox output_report.html &

.PHONY: weights
weights:
	@sed -i 's/\r//' ${CONV_SCRIPT}
	@chmod +x ${CONV_SCRIPT}
	test -f ${MODEL_FILENAME} || (${GET_WEIGHTS} && ${CONV_SCRIPT})

.PHONY: compile
compile: weights
	test -f graph || ${NCCOMPILE} -s 12 ${MODEL_FILENAME} ${INPUT_NODE_FLAG} ${OUTPUT_NODE_FLAG}

.PHONY: check
check: weights
	-${NCCHECK} -s 12 ${MODEL_FILENAME} ${INPUT_NODE_FLAG} ${OUTPUT_NODE_FLAG} -i ../../data/images/cat.jpg -id 917 -M 128 -S 2 -cs 0,1,2

.PHONY: run
run: compile
	./run.py

.PHONY: run_py
run_py: compile
	./run.py

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: prototxt, profile, compile, check, cpp, run_py, run_cpp";
	@echo "  make weights - downloads the trained model";
	@echo "  make compile - runs SDK compiler tool to compile the NCS graph file for the network";
	@echo "  make check - runs SDK checker tool to verify an NCS graph file";
	@echo "  make profile - runs the SDK profiler tool to profile the network creating output_report.html";
	@echo "  make browse_profile - runs the SDK profiler tool and brings up report in browser.";
	@echo "  make run_py - runs the run.py python example program";
	@echo "  make clean - removes all created content"

clean: 
	rm -f output.gv
	rm -f output.gv.svg
	rm -f output_report.html
	rm -f output_expected.npy
	rm -f *.ckpt
	rm -f output_result.npy
	rm -f output_val.csv
	rm -rf output
	rm -f graph
  
