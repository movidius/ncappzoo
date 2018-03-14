.PHONY: all
all: dependencies ilsvrc12 googlenet squeezenet alexnet

.PHONY: dependencies
dependencies:
	@echo "\Installing dependencies"
	sudo apt-get install python3-tk

.PHONY: ilsvrc12
ilsvrc12:
	@echo "\making ilsvrc12"
	(cd ../../data/ilsvrc12; make;)

.PHONY: mobilenet
mobilenet: 
	@echo "\nmaking mobilenet"
	(cd ../../tensorflow/mobilenets; make compile;)

.PHONY: googlenet
googlenet: 
	@echo "\nmaking googlenet"
	(cd ../../caffe/GoogLeNet; make compile;)

.PHONY: squeezenet
squeezenet:
	@echo "\nmaking squeezenet"
	(cd ../../caffe/SqueezeNet; make compile;)

.PHONY: alexnet
alexnet:
	@echo "\nmaking alexet"
	(cd ../../caffe/AlexNet; make compile;)


.PHONY: run
run: all
	@(if [ -f ./inferences.csv ]; then rm inferences.csv; fi )
	@echo "\nRunning log-image-classifier.py";
	python3 log-image-classifier.py

.PHONY: clean
clean:
	@echo "\nmaking clean";
	@echo "  Nothing to clean";

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - Shows this message";
	@echo "  make - Builds all dependencies, but does not run this program";
	@echo "  make run - Runs this program";
