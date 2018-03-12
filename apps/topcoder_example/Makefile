DATADIR = data
TRAINING_DATA = https://www.topcoder.com/contest/problem/IntelMovidius/training.tar
TRAINING_CSV = https://www.topcoder.com/contest/problem/IntelMovidius/training_ground_truth.csv
PROVISIONAL_DATA = https://www.topcoder.com/contest/problem/IntelMovidius/provisional.tar
CATEGORIES_TXT = https://www.topcoder.com/contest/problem/IntelMovidius/categories.txt

.PHONY: default
default: help ;

.PHONY: all
all: help ;

data:
	@(mkdir -p $(DATADIR);)
	@echo "Downloading training dataset..."
	@(cd $(DATADIR); wget $(TRAINING_CSV); wget $(TRAINING_DATA); tar -xvf training.tar;)
	@echo "Downloading provisional dataset..."
	@(cd $(DATADIR); cp ../provisional.csv .; wget $(PROVISIONAL_DATA); tar -xvf provisional.tar;)
	@echo "Downloading labels file..."
	@(cd $(DATADIR); wget $(CATEGORIES_TXT);)

compile:
	@mvNCCompile -s 12 -w weights.caffemodel network.prototxt -o compiled.graph

profile:
	@mvNCProfile -s 12 -w weights.caffemodel network.prototxt

infer:
	@(cd supporting/ && python3 inferences.py $(shell pwd)/$(DATADIR) )

help:
	@(cd supporting/ && python3 inferences.py -h )

zip:
	@zip -r submission.zip compiled.graph inferences.csv network.prototxt weights.caffemodel supporting/

run: data compile infer

clean:
	rm -f output*
	rm -f graph
	rm -f compiled.graph
	rm -f categories.txt
	rm -f inferences.csv

