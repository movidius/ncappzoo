

.PHONY: all
all: compile_googlenet compile_alexnet compile_squeezenet compile_agenet compile_gendernet
	@echo "making all"

.PHONY: run
run: run_googlenet run_alexnet run_squeezenet run_agenet run_gendernet

.PHONY: compile_googlenet
compile_googlenet:
	(cd ../../caffe/GoogLeNet; make compile; cd ../../apps/benchmarkncs;)

.PHONY: compile_alexnet
compile_alexnet:
	(cd ../../caffe/AlexNet; make compile; cd ../../apps/benchmarkncs;)

.PHONY: compile_squeezenet
compile_squeezenet:
	(cd ../../caffe/SqueezeNet; make compile; cd ../../apps/benchmarkncs;)

.PHONY: compile_agenet
compile_agenet:
	(cd ../../caffe/AgeNet; make compile; cd ../../apps/benchmarkncs;)

.PHONY: compile_gendernet
compile_gendernet:
	(cd ../../caffe/GenderNet; make compile; cd ../../apps/benchmarkncs;)


.PHONY: run_googlenet
run_googlenet: compile_googlenet
	@./benchmarkncs.py ../../caffe/GoogLeNet ../../data/images 224 224

.PHONY: run_alexnet
run_alexnet: compile_alexnet
	@./benchmarkncs.py ../../caffe/AlexNet ../../data/images 227 227

.PHONY: run_squeezenet
run_squeezenet: compile_squeezenet
	@./benchmarkncs.py ../../caffe/SqueezeNet ../../data/images 227 227

.PHONY: run_agenet
run_agenet: compile_agenet
	@./benchmarkncs.py ../../caffe/AgeNet ../../data/images 227 227

.PHONY: run_gendernet
run_gendernet: compile_gendernet
	@./benchmarkncs.py ../../caffe/GenderNet ../../data/images 227 227

.PHONY: help
help:
	@echo "Possible make targets:"
	@echo "  make help: Shows this message."
	@echo "  make all: Makes everything needed to run the program."
	@echo "  make run: Runs the program for googlenet, alexnet, squeezenet, gendernet, agenet"
	@echo "  make run_googlenet: Runs the program for GoogLeNet"
	@echo "  make run_alexnet: Runs the program for AlexNet"
	@echo "  make run_squeezenet: Runs the program for SqueezeNet"
	@echo "  make run_agenet: Runs the program for AgeNet"
	@echo "  make run_gendernet: Runs the program for GenderNet"
	@echo "  make clean: Removes all temp files and build products created in this directory." 


.PHONY: clean
clean:
	@echo "making clean"

