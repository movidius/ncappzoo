python_version_full := $(wordlist 2,4,$(subst ., ,$(shell python3 --version 2>&1)))
python_version_minor := $(word 2,${python_version_full})

NCCOMPILE = /usr/local/bin/ncCompile
NCPROFILE = /usr/local/bin/ncProfile
NCCHECK = /usr/local/bin/ncCheck

PROTOTXT_FILENAME= deploy.prototxt
GET_PROTOTXT =  wget -P . https://github.com/DeepScale/SqueezeNet/raw/master/SqueezeNet_v1.0/deploy.prototxt

CAFFEMODEL_FILENAME = squeezenet_v1.0.caffemodel
GET_CAFFEMODEL = wget -P . -N https://github.com/DeepScale/SqueezeNet/raw/master/SqueezeNet_v1.0/squeezenet_v1.0.caffemodel

.PHONY: all
all: profile compile run

.PHONY: py
py:
	@echo ${python_version_minor}

.PHONY: prereqs
prereqs:
	@(cd ../../data/ilsvrc12; make)
	@test -f LICENSE.sqeezenet || (wget -P . -N https://github.com/DeepScale/SqueezeNet/raw/master/LICENSE;mv LICENSE LICENSE.squeezenet)

.PHONY: prototxt
prototxt: prereqs
	@test -f ${PROTOTXT_FILENAME} || (${GET_PROTOTXT}; sed -i '11s/10/1/' ${PROTOTXT_FILENAME}; sed -i '530s/pad/#pad/' ${PROTOTXT_FILENAME};)

.PHONY: profile
profile: prototxt
	@${NCPROFILE} ${PROTOTXT_FILENAME} -s 12
	@firefox output_report.html &

.PHONY: compile
compile: prototxt
	@test -f ${CAFFEMODEL_FILENAME} || ${GET_CAFFEMODEL}
	@${NCCOMPILE} -w ${CAFFEMODEL_FILENAME} -s 12 ${PROTOTXT_FILENAME}

.PHONY: check
check: prototxt
	@test -f ${CAFFEMODEL_FILENAME} || ${GET_CAFFEMODEL}
	@${NCCHECK} -w ${CAFFEMODEL_FILENAME} -i images/cat.jpg -s 12 -id 885 ${PROTOTXT_FILENAME}

.PHONY: run
run:
	./run.py


cleancaffemodel:
	rm -f ${PROTOTXT_FILENAME}
	rm -f ${CAFFEMODEL_FILENAME}

clean: cleancaffemodel
	rm -f graph
	rm -f output.gv
	rm -f output.gv.svg
	rm -f output_report.html
	rm -f output_expected.npy
	rm -f zero_weights.caffemodel
	rm -f output_result.npy
	rm -f output_val.csv
	rm -f LICENSE.squeezenet
