
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif


# filenames for the graph files that we'll copy to this directory.
MNIST_GRAPH_FILENAME = mnist_inference.graph

.PHONY: all
all: deps


.PHONY: deps
deps:
	@echo "\nmaking deps"
	(cd ../../tensorflow/mnist; make compile; cd ../../apps/mnist_calc; cp ../../tensorflow/mnist/${MNIST_GRAPH_FILENAME} ./${MNIST_GRAPH_FILENAME};)


.PHONY: data
data:
	@echo "\nmaking data - no data needed."
	

.PHONY: run
run: deps
	@echo "\nmaking run"
	python3 ./touchcalc.py


.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes everything needed to run but doesn't run";
	@echo "  make clean - removes all created content in this directory."
	@echo "  make data - downloads data as needed"
	@echo "  make deps - makes/prepares dependencies"
	@echo "  make run or run_py - runs the birds.py python example program";


.PHONY: clean
clean: 
	@echo "\nmaking clean"
	rm -f ${MNIST_GRAPH_FILENAME}


