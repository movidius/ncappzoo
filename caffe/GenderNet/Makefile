
NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

PROTOTXT_FILENAME= deploy.prototxt
GET_PROTOTXT =  wget -P . https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net_definitions/deploy.prototxt

CAFFEMODEL_FILENAME = gender_net.caffemodel
GET_CAFFEMODEL = wget -P . -N https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel

.PHONY: all
all: compile

.PHONY: prereqs
prereqs:
	@(cd ../../data/age_gender; make)

.PHONY: prototxt
prototxt: prereqs
	test -f ${PROTOTXT_FILENAME} || (${GET_PROTOTXT}; awk 'NR <2 {print}' < ${PROTOTXT_FILENAME} > temp; cat input_shape.prototxt >> temp; awk 'NR > 6 {print}' < ${PROTOTXT_FILENAME} >> temp; mv temp ${PROTOTXT_FILENAME})

.PHONY: profile
profile: prototxt
	${NCPROFILE} ${PROTOTXT_FILENAME} -s 12
#	firefox output_report.html &

.PHONY: caffemodel
caffemodel: prototxt
	test -f ${CAFFEMODEL_FILENAME} || ${GET_CAFFEMODEL}

.PHONY: compile
compile: prototxt
	test -f ${CAFFEMODEL_FILENAME} || ${GET_CAFFEMODEL}
	${NCCOMPILE} -w ${CAFFEMODEL_FILENAME} -s 12 ${PROTOTXT_FILENAME}

.PHONY: check
check: prototxt
	test -f ${CAFFEMODEL_FILENAME} || ${GET_CAFFEMODEL}
	${NCCHECK} -w ${CAFFEMODEL_FILENAME} -i ../../data/images/cat.jpg -s 12 -id 885 ${PROTOTXT_FILENAME}

.PHONY: run
run: compile
	python3 ./run.py

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes everything needed to run";
	@echo "  make prototxt - downloads and adds input shape to Caffe prototxt file";
	@echo "  make caffemodel - downloads the caffemodel for the network"
	@echo "  make compile - runs SDK compiler tool to compile the NCS graph file for the network";
	@echo "  make check - runs SDK checker tool to verify an NCS graph file";
	@echo "  make profile - runs the SDK profiler tool to profile the network creating output_report.html";
	@echo "  make run - runs the run.py python example program:";
	@echo "  make clean - removes all created content"


cleancaffemodel:
	@rm -f ${PROTOTXT_FILENAME}
	@rm -f ${CAFFEMODEL_FILENAME}

clean: cleancaffemodel
	@rm -f graph
	@rm -f output.gv
	@rm -f output.gv.svg
	@rm -f output_report.html
	@rm -f output_expected.npy
	@rm -f zero_weights.caffemodel
	@rm -f output_result.npy
	@rm -f output_val.csv
	@rm -f image.jpg
