


.PHONY: all
all: multistick_cpp

.PHONY: googlenet
googlenet: 
	@echo "\nmaking googlenet"
	(cd ../../caffe/GoogLeNet; make compile; cd ../../apps/multistick_cpp; cp ../../caffe/GoogLeNet/graph ./googlenet.graph;)

.PHONY: squeezenet
squeezenet: 
	@echo "\nmaking squeezenet"
	(cd ../../caffe/SqueezeNet; make compile; cd ../../apps/multistick_cpp; cp ../../caffe/SqueezeNet/graph ./squeezenet.graph;)

.PHONY:	multistick_cpp
multistick_cpp: googlenet squeezenet
	@echo "\nmaking multistick_cpp"
	cp googlenet.graph cpp/googlenet.graph;
	cp squeezenet.graph cpp/squeezenet.graph;
	g++ cpp/multistick.cpp cpp/fp16.c -o cpp/multistick_cpp -lmvnc
	@echo "Created cpp/multistick_cpp executable"

.PHONY: run
run: multistick_cpp
	@echo "\nmaking run";
	cd cpp; ./multistick_cpp; cd ..;

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: multistick_cpp, run";
	@echo "  make multistick_cpp - builds the multistick_cpp executable example";
	@echo "  make run - runs the multistick_cpp executable example program";
	@echo "  make clean - removes all created content and temporary files";

clean: clean
	@echo "\nmaking clean";
	rm -f cpp/multistick_cpp;
	rm -f cpp/googlenet.graph;
	rm -f cpp/squeezenet.graph;
	rm -f squeezenet.graph;
	rm -f googlenet.graph;

