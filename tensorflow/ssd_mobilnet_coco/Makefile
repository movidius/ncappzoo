
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

SSD_V1_WEIGHTS_FILENAME = ssd_mobilenet_v1_coco_2018_01_28.tar.gz
SSD_V2_WEIGHTS_FILENAME = ssd_mobilenet_v2_coco_2018_03_29.tar.gz

GET_V1_WEIGHTS = (wget http://download.tensorflow.org/models/object_detection/${SSD_V1_WEIGHTS_FILENAME} && tar zxf ${SSD_V1_WEIGHTS_FILENAME} && rm ${SSD_V1_WEIGHTS_FILENAME})
GET_V2_WEIGHTS = (wget http://download.tensorflow.org/models/object_detection/${SSD_V2_WEIGHTS_FILENAME} && tar zxf ${SSD_V2_WEIGHTS_FILENAME} && rm ${SSD_V2_WEIGHTS_FILENAME})

V1_MODEL_FILENAME = ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb
V2_MODEL_FILENAME = ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb

SSD_CONFIG_FILE = ssd_mob.config

.PHONY: all
all: compile_v1 run_v1

.PHONY: prereqs
prereqs:
	(cd ../../data/ilsvrc12; make)
	@sed -i 's/\r//' run.py
	@chmod +x run.py

.PHONY: weights_v1
weights_v1:
	test -f ${V1_MODEL_FILENAME} || (${GET_V1_WEIGHTS})

.PHONY: weights_v2
weights_v2:
	test -f ${V2_MODEL_FILENAME} || (${GET_V2_WEIGHTS})

.PHONY: compile_v1
compile_v1: weights_v1
	test -f ssd_mob_v1_graph || ${NCCOMPILE} -s 12 ${V1_MODEL_FILENAME} --tf-ssd-config ${SSD_CONFIG_FILE} -o ssd_mob_v1_graph

.PHONY: compile_v2
compile_v2: weights_v2
	test -f ssd_mob_v2_graph || ${NCCOMPILE} -s 12 ${V2_MODEL_FILENAME} --tf-ssd-config ${SSD_CONFIG_FILE} -o ssd_mob_v2_graph


.PHONY: run
run: compile_v1
	python3 run.py

.PHONY: run_v1
run_v1: compile_v1
	python3 run.py

.PHONY: run_v2
run_v2: compile_v2
	python3 run.py --ncsgraph ssd_mob_v2_graph
	
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
	rm -f ssd_mob_v1_graph
	rm -f ssd_mob_v2_graph
	rm -rf ssd_mobilenet_v1_coco_2018_01_28
	rm -rf ssd_mobilenet_v2_coco_2018_03_29

