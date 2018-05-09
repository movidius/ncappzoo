.PHONY: all
all: ilsvrc12 googlenet

.PHONY: ilsvrc12
ilsvrc12:
	@echo "\making ilsvrc12"
	(cd ../../data/ilsvrc12; make;)

.PHONY: gendernet
gendernet: 
	@echo "\nmaking gendernet"
	(cd ../../caffe/GenderNet; make compile;)

.PHONY: agenet
agenet: 
	@echo "\nmaking AgeNet"
	(cd ../../caffe/AgeNet; make compile;)

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
	@echo "\nRunning image-classifier.py";
	python3 image-classifier.py

 .PHONY: clean
 clean: 
	@echo "\nmaking clean"
	@echo "  Nothing to remove.";

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - Shows this message";
	@echo "  make - Builds all dependencies, but does not run this program";
	@echo "  make run - Runs this program";
	@echo "  make clean - removes files files and directories created in this directory";

