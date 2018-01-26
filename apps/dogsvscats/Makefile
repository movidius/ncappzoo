DATA_ROOT?=data
CAFFE_PATH?=~/workspace/caffe
TRAIN_DATA_ROOT=$(DATA_ROOT)/train
TRAIN_LMDB_ROOT=$(DATA_ROOT)/dogsvscats_train_lmdb
VAL_LMDB_ROOT=$(DATA_ROOT)/dogsvscats_val_lmdb

.PHONY: all
all: deps unzip labels lmdb mean

.PHONY: deps
deps:
	@echo "Installing dependencies..."
	sudo apt-get -y install unzip 
	sudo pip install pyyaml

.PHONY: unzip
unzip:
	@echo "Extracting dataset..."
	@( unzip $(DATA_ROOT)/test1.zip -d $(DATA_ROOT) )
	@( unzip $(DATA_ROOT)/train.zip -d $(DATA_ROOT) )

.PHONY: labels
labels:
	@echo "Creating labels file..."
	@( python create-labels.py $(DATA_ROOT) )
	@echo "Done. See $(DATA_ROOT)/train/train.txt & val.txt"

.PHONY: lmdb
lmdb:
	echo "Creating lmdb files..."
	@( ./create-lmdb.sh $(DATA_ROOT) )
	@echo "Done. See $(DATA_ROOT)/dogsvscats_train_lmdb & dogsvscats_val_lmdb"

.PHONY: mean 
mean:
	@echo "Computing image mean..."
	@($(CAFFE_PATH)/build/tools/compute_image_mean $(TRAIN_LMDB_ROOT) $(DATA_ROOT)/dogsvscats-train-mean.binaryproto)
	@($(CAFFE_PATH)/build/tools/compute_image_mean $(VAL_LMDB_ROOT) $(DATA_ROOT)/dogsvscats-val-mean.binaryproto)
	@echo "Done. See $(DATA_ROOT)/dogsvscats-train-mean.binaryproto & dogsvscats-val-mean.binaryproto"

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - Runs all targets in the required sequence";
	@echo "  make deps - Installs dependencies";
	@echo "  make unzip - Extracts dataset";
	@echo "  make labels - Creates train.txt & val.txt needed to generate lmdb file";
	@echo "  make lmdb - Creates lmdb files needed to initiate Caffe training";
	@echo "  make mean - Computes dataset mean";

.PHONY: clean
clean:
	@echo "making clean"
	rm -rf $(DATA_ROOT)/train
	rm -rf $(DATA_ROOT)/test1
	rm -rf $(DATA_ROOT)/dogsvscats_train_lmdb
	rm -rf $(DATA_ROOT)/dogsvscats_val_lmdb
	rm -rf $(DATA_ROOT)/dogsvscats-train-mean.binaryproto
	rm -rf $(DATA_ROOT)/dogsvscats-val-mean.binaryproto
