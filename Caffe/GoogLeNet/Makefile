python_version_full := $(wordlist 2,4,$(subst ., ,$(shell python3 --version 2>&1)))
python_version_minor := $(word 2,${python_version_full})

NCCOMPILE = /usr/local/bin/ncCompile
NCPROFILE = /usr/local/bin/ncProfile
NCCHECK = /usr/local/bin/ncCheck

PROTOTXT_FILENAME= googlenet.prototxt
CAFFEMODEL_FILENAME = bvlc_googlenet.caffemodel
GET_CAFFEMODEL = wget -P . -N http://dl.caffe.berkeleyvision.org/${CAFFEMODEL_FILENAME}

.PHONY: all
all: profile compile run

.PHONY: py
py:
	@echo ${python_version_minor}

.PHONY: profile
profile: 
	${NCPROFILE} googlenet.prototxt -s 12
	firefox output_report.html &

.PHONY: compile
compile:
	test -f ${CAFFEMODEL_FILENAME} || ${GET_CAFFEMODEL}
	${NCCOMPILE} -w ${CAFFEMODEL_FILENAME} -s 12 ${PROTOTXT_FILENAME}

.PHONY: check
check:
	test -f ${CAFFEMODEL_FILENAME} || ${GET_CAFFEMODEL}
	${NCCHECK} -w ${CAFFEMODEL_FILENAME} -i images/cat.jpg -s 12 -id 885 ${PROTOTXT_FILENAME}

.PHONY: run
run:
	./run.py


cleancaffemodel:
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
