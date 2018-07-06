
MODEL_CKPT_ZIP=submission_andresuduque.zip

YELLOW='\033[1;33m'
NOCOLOR='\033[0m'

all: deps compile data

deps:
	@echo $(YELLOW)"\ndeps: Downloading submission archive ..."$(NOCOLOR)
	test -f ./submission_andresuduque.zip || (./download-submission.sh);
	@echo $(YELLOW)"      Uncompressing model"$(NOCOLOR)
	@unzip -o -d submission $(MODEL_CKPT_ZIP);
	@mv ./submission/supporting/trained.zip .
	@unzip -o trained.zip;
#	@echo $(YELLOW)"\nRemoving temporary files ... "$(NOCOLOR);
#	@rm -f submission_andresuduque.zip;
	@rm -f trained.zip;

compile: deps
	@echo $(YELLOW)"\nCompiling model for NCS ..."$(NOCOLOR);
	mvNCCompile ./output/network.meta -in=input -on=final_result -o compiled.graph -s12;

profile: deps
	mvNCProfile ./output/network.meta -in=input -on=final_result -is 299 299 -s12;

check: deps
	mvNCCheck ./output/network.meta -in=input -on=final_result -is 299 299 -s12;

data:
	@echo $(YELLOW)"\nDownloading test images ..."$(NOCOLOR);
	test -f ./provisional.tar || (wget https://www.topcoder.com/contest/problem/IntelMovidius/provisional.tar);
	@echo $(YELLOW)"Uncompressing images ..."$(NOCOLOR);
	@tar xf provisional.tar;
#	@echo $(YELLOW)"Removing temporary files ... "$(NOCOLOR);
#	@rm provisional.tar;

run: deps data compile
	@echo $(YELLOW)"\nRunning inferences ..."$(NOCOLOR);
	python3 ./supporting/inferences.py -images-dir ./provisional -output-file ./inferences.csv -graph-file ./compiled.graph -labels-map-file ./supporting/labels.txt

help:
	@echo "Possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: deps, compile, data, run";
	@echo "  make deps - downloads the trained model";
	@echo "  make compile - runs SDK compiler tool to compile the NCS graph file for the network";
	@echo "  make check - runs SDK checker tool to verify an NCS graph file";
	@echo "  make profile - runs the SDK profiler tool to profile the network creating output_report.html";
	@echo "  make data - downloads the Intel Movidius Marathon Match test dataset";
	@echo "  make run - runs inferences on the test dataset";
	@echo "  make clean - removes additionally created content";

clean: 
	rm -rf ./submission
	rm -rf ./provisional
	rm -rf ./output
	rm -f submission_andresuduque.zip
	rm -f provisional.tar
	rm -f compiled.graph
	rm -f graph cookie
	rm -f output*
	rm -f inferences.csv
