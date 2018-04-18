
YELLOW='\033[1;33m'
NOCOLOR='\033[0m'
VIDEO?=0

.PHONY: all
all: run

.PHONY: mobilenet-ssd
mobilenet-ssd: 
	@echo "\n================================================================"
	@echo $(YELLOW)"\nMaking MobileNet SSD..."$(NOCOLOR);
	@(cd ../../caffe/SSD_MobileNet; test -f graph || make compile;)
	@echo $(YELLOW)"Done! See ../../caffe/SSD_MobileNet/graph.\n"$(NOCOLOR)

.PHONY: run
run: mobilenet-ssd
	@echo "\n================================================================"
	@echo $(YELLOW)"\nRunning security-cam.py..."$(NOCOLOR);
	python3 security-cam.py --video $(VIDEO)

.PHONY: clean
clean:
	@echo "\n================================================================"
	@echo $(YELLOW)"\nRunning clean..."$(NOCOLOR);
	@(rm -rf captures/photo*)
	@echo $(YELLOW)"Done! Deleted all snapshots.\n"$(NOCOLOR)

.PHONY: help
help:
	@echo "\n================================================================"
	@echo $(YELLOW)"possible make targets: "$(NOCOLOR);
	@echo "  make help - Shows this message";
	@echo "  make - Builds all dependencies, but does not run this program";
	@echo "  make run - Runs this program";
	@echo "  make clean - Delete snapshots";

