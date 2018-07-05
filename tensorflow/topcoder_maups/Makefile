
all: deps compile data build run

deps:
	@echo "deps: downloading model";
	@./download_model.sh;
	@echo "      unconpressing model";
	@tar -xzf model.tgz;
	@echo "      removing temporary files";
	@rm model.tgz;
	@echo "      done";
	@echo "";

compile:
	@echo "compile: compiling model for NCS";
	@mvNCCompile model_inceptionv4.meta -w model_inceptionv4 -s 12 -in input -on output -is 299 299 -o compiled.graph > /dev/null 2> /dev/null;
	@echo "         done";
	@echo "";

profile:
	mvNCProfile -s 12 model_inceptionv4.meta -in input -on output

check:
	mvNCCheck -s 12 model_inceptionv4.meta -in input -on output -i image.jpg -id 26 -S 2 -M 1 -cs 2,1,0

data:
	@echo "data: downloading test images";
	@wget -q https://www.topcoder.com/contest/problem/IntelMovidius/provisional.tar
	@echo "      unconpressing images";
	@tar xf provisional.tar;
	@echo "      removing temporary files";
	@rm provisional.tar;
	@echo "      done";
	@echo "";

build:
	@echo "build: building example";
	@g++ -O3 -std=c++11 inference.cpp fp16.c -lmvnc `pkg-config --libs --cflags opencv` -o inference;
	@echo "       done";
	@echo "";

run:
	@echo "run: running example";
	@./inference > inferences.csv
	@echo "     done";
	@echo "";

help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "";
	@echo "  make all - makes the following: deps, compile, data, build, run";
	@echo "";
	@echo "  make deps - downloads the trained model";
	@echo "  make compile - runs SDK compiler tool to compile the NCS graph file for the network";
	@echo "  make check - runs SDK checker tool to verify an NCS graph file";
	@echo "  make profile - runs the SDK profiler tool to profile the network creating output_report.html";
	@echo "";
	@echo "  make data - downloads the Intel Movidius Marathon Match test dataset";
	@echo "  make build - build the example program";
	@echo "  make run - runs the example program";
	@echo "";
	@echo "  make clean - removes all created content";

clean: 
	@rm -f model_inceptionv4.* model.tgz
	@rm -f compiled.graph
	@rm -f graph output_expected.npy output_report.html output.gv output.gv.svg
	@rm -f output_result.npy output_val.csv
	@rm -rf provisional provisional.tar
	@rm -f inference
	@rm -f inferences.csv
