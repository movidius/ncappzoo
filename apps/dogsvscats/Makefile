CAFFE_SRC_PATH?=~/workspace/caffe
DATA_DIR?=data
TRAIN_DATA_DIR=$(DATA_DIR)/train
TRAIN_LMDB_DIR=$(DATA_DIR)/train_lmdb
VAL_LMDB_DIR=$(DATA_DIR)/val_lmdb

RED='\033[1;33m'
YELLOW='\033[1;33m'
NOCOLOR='\033[0m'

.PHONY: all
all: profile

check_caffe_src:
ifndef CAFFE_SRC_PATH
	$(error CAFFE_SRC_PATH is not defined. Run `export CAFFE_SRC_PATH=path/to/your/caffe/source/repo`)
endif

.PHONY: extract_data
extract_data:
	@echo "\n===================================================================="
	@echo $(YELLOW)"Extracting dataset..."$(NOCOLOR)
	@( cd $(DATA_DIR); test -d "test1" || unzip test1.zip )
	@( cd $(DATA_DIR); test -d "train" || unzip train.zip )
	@echo $(YELLOW)"Done. See $(DATA_DIR)/test1 & train folders."$(NOCOLOR)

.PHONY: labels
labels: extract_data
	@echo "\n===================================================================="
	@echo $(YELLOW)"Creating labels file..."$(NOCOLOR)
	@( python create-labels.py --data_dir $(DATA_DIR) )
	@echo $(YELLOW)"Done. See $(DATA_DIR)/train.txt & val.txt."$(NOCOLOR)

.PHONY: lmdb
lmdb: check_caffe_src labels
	@echo "\n===================================================================="
	@echo $(YELLOW)"Creating lmdb files..."$(NOCOLOR)
	@( ./create-lmdb.sh $(DATA_DIR) )
	@echo $(YELLOW)"Done. See $(DATA_DIR)/train_lmdb & val_lmdb."$(NOCOLOR)

.PHONY: mean 
mean: check_caffe_src lmdb
	@echo "\n===================================================================="
	@echo $(YELLOW)"Computing image mean..."$(NOCOLOR)
	@($(CAFFE_SRC_PATH)/build/tools/compute_image_mean $(TRAIN_LMDB_DIR) $(DATA_DIR)/train-mean.binaryproto 2>&1 | tee $(DATA_DIR)/train-mean.txt)
	@($(CAFFE_SRC_PATH)/build/tools/compute_image_mean $(VAL_LMDB_DIR) $(DATA_DIR)/val-mean.binaryproto 2>&1 | tee $(DATA_DIR)/val-mean.txt)
	@echo $(YELLOW)"Done. See $(DATA_DIR)/train-mean.binaryproto, train-mean.txt, val-mean.binaryproto & val-mean.txt"$(NOCOLOR)

.PHONY: download_weights
download_weights: 
	@echo "\n===================================================================="
	@echo $(YELLOW)"Downloading pre-trained weights..."$(NOCOLOR)
	@($(CAFFE_SRC_PATH)/scripts/download_model_binary.py $(CAFFE_SRC_PATH)/models/bvlc_googlenet)
	@echo $(YELLOW)"Done. Look for .caffemodel in $(CAFFE_SRC_PATH)/models/bvlc_googlenet."$(NOCOLOR)

.PHONY: train
train: mean
	@echo "\n===================================================================="
	@echo $(YELLOW)"Training the model..."$(NOCOLOR)
	($(CAFFE_SRC_PATH)/build/tools/caffe train --solver bvlc_googlenet/org/solver.prototxt 2>&1 | tee bvlc_googlenet/org/train.log)
	@echo $(YELLOW)"Done. Look for .caffemodel files in bvlc_googlenet/org."$(NOCOLOR)

.PHONY: compile
compile:
	@echo "\n===================================================================="
	@echo $(YELLOW)"\nCompiling model to Movidius graph..."$(NOCOLOR)
	(cd bvlc_googlenet/org; mvNCCompile -s 12 deploy.prototxt -w bvlc_googlenet_iter_40000.caffemodel;)

.PHONY: profile
profile:
	@echo "\n===================================================================="
	@echo $(YELLOW)"\nProfiling the model..."$(NOCOLOR)
	(cd bvlc_googlenet/org; mvNCProfile -s 12 deploy.prototxt -w bvlc_googlenet_iter_40000.caffemodel;)

.PHONY: check
check:
	@echo "\n===================================================================="
	@echo $(YELLOW)"\nComparing results with standard Tensorflow..."$(NOCOLOR)
	(cd bvlc_googlenet/org; mvNCCheck -s 12 deploy.prototxt -w bvlc_googlenet_iter_40000.caffemodel;)

.PHONY: run
run:
	@echo "\n===================================================================="
	@echo $(YELLOW)"\nRunning inferences using image-classifier project..."$(NOCOLOR)
	(python3 ../../apps/rapid-image-classifier/rapid-image-classifier.py --graph ./bvlc_googlenet/org/graph --dim 224 224 --mean 104.00698793 116.66876762 122.67891434 --scale 1 --labels ./data/categories.txt --image ./data/test1/)

.PHONY: help
help:
	@echo "\n===================================================================="
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - Runs all targets in the required sequence";
	@echo "  make extract_data - Extracts training & test images downloaded from Kaggle";
	@echo "  make labels - Creates train.txt & val.txt needed to generate lmdb file";
	@echo "  make lmdb - Creates lmdb files needed to initiate Caffe training";
	@echo "  make mean - Computes dataset mean";
	@echo "  make compile - Convert the trained model into Movidius graph file.";
	@echo "  make profile - Run the model on NCS and extract complexity, bandwidth and execution time for each layer.";
	@echo "  make check - Compare the results of running this model on the NCS vs on standard Caffe GPU/CPU.";
	@echo "  make run - Run inference using rapid-image-classifier app.";

.PHONY: clean
clean:
	@echo "Deleting processed data files..."
	@rm -rf data/train.txt data/val.txt data/train-mean.txt data/val-mean.txt
	@rm -rf data/train data/train_lmdb
	@rm -rf data/test1 data/val_lmdb
	@rm -rf data/*.binaryproto
