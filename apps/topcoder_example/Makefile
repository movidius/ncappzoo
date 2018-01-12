MOVIDIUSDIR = /path/to/movidius/folder

.PHONY: default
default: help ;

compile:
	@mvNCCompile -s 12 -w weights.caffemodel network.prototxt -o compiled.graph

profile:
	@mvNCProfile -s 12 -w weights.caffemodel network.prototxt

infer:
	@(cd supporting/ && python3 inferences.py $(MOVIDIUSDIR) )

help:
	@(cd supporting/ && python3 inferences.py -h )

zip:
	@zip -r submission.zip compiled.graph inferences.csv network.prototxt weights.caffemodel supporting/
