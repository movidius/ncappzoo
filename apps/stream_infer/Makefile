
.PHONY: all
all: squeezenet

.PHONY: squeezenet
squeezenet: 
	@echo "\nmaking squeezenet"
	(cd ../../caffe/SqueezeNet; make compile;)

.PHONY: run
run: squeezenet
	@echo "\nmaking run";
	python3 stream_infer.py

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: run";
	@echo "  make run - runs the stream_infer program";
	@echo "  make clean - removes all created content and temporary files";

clean: clean
	@echo "\nmaking clean";


