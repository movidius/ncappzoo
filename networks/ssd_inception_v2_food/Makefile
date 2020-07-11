NCCOMPILE = mo.py
CPU_EXTENSION = $(INTEL_OPENVINO_DIR)/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so

YELLOW='\033[1;33m'
NOCOLOR='\033[0m'
RED = '\033[1;31m'

GRAPH_FILENAME_BASE_IN_DIR_FP16 = frozen_inference_graph.xml
MO_LABELS = frozen_inference_graph.labels

MO_INPUT_GRAPH_ARG = --input_model frozen_inference_graph.pb
MO_INPUT_PIPELINE = --tensorflow_object_detection_api_pipeline_config ssd_inception_v2_coco.config
MO_INPUT_CUSTOM_OPS = --tensorflow_use_custom_operations_config ssd_v2_support_inception.json

MO_OUTPUT_DIR_FP16 = --output_dir ssd_inception_v2_mo_fp16

MODEL_PB = frozen_inference_graph.pb
MODEL_INFERENCE_PIPELINE_CONFIG = ssd_inception_v2_coco.config
MODEL_INFERENCE_PB = frozen_inference_graph.pb

GET_TEST_IMG1 = wget -c --no-cache -P ./test_images https://raw.githubusercontent.com/fcr3/chef_devmesh/master/test_images/test1.JPG
GET_TEST_IMG2 = wget -c --no-cache -P ./test_images https://raw.githubusercontent.com/fcr3/chef_devmesh/master/test_images/test2.JPG
GET_TEST_IMG3 = wget -c --no-cache -P ./test_images https://raw.githubusercontent.com/fcr3/chef_devmesh/master/test_images/test3.JPG
GET_TEST_LICENSE = wget -c --no-cache -P ./test_images https://raw.githubusercontent.com/fcr3/chef_devmesh/master/test_images/license.txt
GET_MODEL_PB = wget -c --no-cache 'https://downloadmirror.intel.com/29611/eng/food_detection_model_labels.zip'

RUN_PY_RELATIVE_DIR = ssd_inception_v2_food.py
TEST_IMG_DIR = test_images
TEST_IMG1 = test_images/test1.JPG
TEST_IMG2 = test_images/test2.JPG
TEST_IMG3 = test_images/test3.JPG
TEST_LICENSE = test_images/license.txt

TRAIN_DIR = train_files
TRAIN_CHKPT_DIR = ${TRAIN_DIR}/ssd_inception_v2_coco_2018_01_28
PRE_TRAINED_MODEL = wget -c --no-cache -P ./${TRAIN_DIR} http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
TRAIN_RECORD_FILE = ./${TRAIN_DIR}/train.record
TRAIN_RECORD = wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Qy8q9b4mC4PuStpAdw6Fq--ao7wY8Y-N' -O ${TRAIN_RECORD_FILE}
TEST_RECORD_FILE = ./${TRAIN_DIR}/test.record
TEST_RECORD = wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zXzehC3S6WLTQbhVOhTOeH3U0xXYM8jb' -O ${TEST_RECORD_FILE}
TRAIN_PBTXT_FILE = ./${TRAIN_DIR}/pascal_label_map.pbtxt
TRAIN_PBTXT = wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NCcklA0Ya-KrSTQi8LHBJUYS5YkHQBKB' -O ${TRAIN_PBTXT_FILE}
SSD_CONFIG_FILE = ./${TRAIN_DIR}/ssd_inception_v2_coco.config
SSD_CONFIG = wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sgDaQgontvm7V9nSdEsF_z2eGJO7kBK5' -O ${SSD_CONFIG_FILE}
TRAIN_PIPE_CONFIG_FILE = ./${TRAIN_DIR}/pipeline.config
TRAIN_PIPE_CONFIG = wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=15FDl27g5wUi4ro7MFIAuCsGxbH-XbSUc' -O ${TRAIN_PIPE_CONFIG_FILE}
JPEG_ZIP_FILE = ./${TRAIN_DIR}/JPEGImages.zip
JPEG_DOWNLOAD = wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=113g9NlQHmORnVWjohgnDPqW26-2Emvjz' -O ${JPEG_ZIP_FILE}
ANNO_ZIP_FILE = ./${TRAIN_DIR}/Annotations.zip
ANNO_DOWNLOAD = wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lYOLT8IuWJ-4osNXPiWeab4GkNg_qEDw' -O ${ANNO_ZIP_FILE}

NETWORK_NAME = SSD Inception v2: Grocery Model
TF_MODEL_DIRNAME = models/
TF_MODEL_REPO = git clone https://github.com/tensorflow/models.git
PROTOC_FILE = models/research/protobuf.zip
PROTOC_DOWNLOAD = wget -O ${PROTOC_FILE} https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip

JPEG_DIRNAME =
ANNOTATIONS_DIRNAME =
MODEL_PIPELINE_FILENAME =
MODEL_CKPT_FILENAME =
PIPELINE_FILENAME =
PBTXT_FILENAME =
TRAIN_REC_FILENAME =
TEST_REC_FILENAME =


.PHONY: all
all: deps data compile_model

.PHONY: deps
deps: get_model
	@echo $(YELLOW)"\nSSD Inception V2 Food: Making dependencies... "$(NOCOLOR)
	@if [ -z "$(INTEL_OPENVINO_DIR)" ] ; \
	then \
		echo "Please initiate the Intel OpenVINO environment by going to the installation directory for openvino and running the setupvars.sh file in the bin folder." ; \
		exit 1 ; \
	else \
		echo "Intel OpenVINO environment is already set!" ; \
	fi

.PHONY: data
data:
	@echo $(YELLOW)"\nSSD Inception V2 Food: Downloading required data..."$(NOCOLOR)
	@if [ -e ${TEST_IMG1} ] ; \
	then \
		echo "test file ${TEST_IMG1} already exists, skipping download"; \
	else \
		echo "Downloading ${TEST_IMG1} file"; \
		${GET_TEST_IMG1}; \
	fi
	@if [ -e ${TEST_IMG2} ] ; \
	then \
		echo "test file ${TEST_IMG2} already exists, skipping download"; \
	else \
		echo "Downloading ${TEST_IMG2} file"; \
		${GET_TEST_IMG2}; \
	fi
	@if [ -e ${TEST_IMG3} ] ; \
	then \
		echo "test file ${TEST_IMG3} already exists, skipping download"; \
	else \
		echo "Downloading ${TEST_IMG3} file"; \
		${GET_TEST_IMG3}; \
	fi
	@if [ -e ${TEST_LICENSE} ] ; \
	then \
		echo "license file ${TEST_LICENSE} already exists, skipping download"; \
	else \
		echo "Downloading ${TEST_LICENSE} file"; \
		${GET_TEST_LICENSE}; \
	fi


.PHONY: get_model
get_model:
	@echo $(YELLOW)"\nSSD Inception V2 Food: Downloading model files..."$(NOCOLOR)
	@if [ -e ${MODEL_PB} ] ; \
	then \
		echo "model file ${MODEL_PB} already exists, skipping download"; \
	else \
		echo "Downloading ${MODEL_PB} file" ; \
		${GET_MODEL_PB} ; \
		unzip food_detection_model_labels.zip ; \
		if [ -e ${MODEL_PB} ] ; \
		then \
			echo "download ${MODEL_PB} done."; \
		else \
			echo "***\nError - Could not download ${MODEL_PB}. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi


.PHONY: compile_model
compile_model: get_model
	@if [ -z "$(INTEL_OPENVINO_DIR)" ] ; \
	then \
		echo "Please initiate the Intel OpenVINO environment by going to the installation directory for openvino and running the setupvars.sh file in the bin folder." ; \
		exit 1 ; \
	else \
		echo "mo.py path is already set!" ; \
	fi
	@echo $(YELLOW)"\nSSD Inception V2 Food: Compiling model to IR..."$(NOCOLOR)
	@if [ -e ${GRAPH_FILENAME_BASE_IN_DIR_FP16} ] ; \
	then \
		echo "compiled FP16 model file already exists, skipping compile."; \
	else \
		echo "Compiling FP16 model..."; \
		${NCCOMPILE} --data_type=FP16 --reverse_input_channels ${MO_INPUT_GRAPH_ARG} ${MO_INPUT_PIPELINE} ${MO_INPUT_CUSTOM_OPS} || (echo $(RED)"Make sure to set the OpenVINO environment variables using the "$(YELLOW)"setupvars.sh"$(RED)" script found in <your OpenVINO install location>/bin/ folder."$(NOCOLOR);  exit 1;); \
	fi


.PHONY: run_FP16
run_FP16: deps data compile_model
	@echo $(YELLOW) "\nSSD Inception V2 Food: Running Python sample..." $(NOCOLOR)
	python3 $(RUN_PY_RELATIVE_DIR) -i ${TEST_IMG1} -m ${GRAPH_FILENAME_BASE_IN_DIR_FP16} -l ${MO_LABELS} -d MYRIAD


.PHONY: run
run: run_py


.PHONY: run_py
run_py: deps data compile_model run_FP16


.PHONY: install-reqs
install-reqs:
	@echo $(YELLOW)"\nSSD Inception V2 Food: Checking application requirements...\n"$(NOCOLOR)
	@echo "No requirements needed."


.PHONY: uninstall-reqs
uninstall-reqs: 
	@echo $(YELLOW)"\nSSD Inception V2 Food: Uninstalling requirements..."$(NOCOLOR)
	@echo "Nothing to uninstall."
	

.PHONY: compile_from_training
compile_from_training:
	@echo $(YELLOW) '\n'${NETWORK_NAME}": Information disclosure before compiling model." $(NOCOLOR);
	@echo "The following steps will be taken in order to compile your own model that detects produce:"
	@echo "  - Freeze trained model"
	@echo "  - Run frozen model through model optimizer"
	@echo ""
	@echo $(YELLOW)"Before proceeding, edit the compile_from_training script in the Makefile."
	@echo "Specifically, edit the trained_checkpoin_prefix to reflect your true model checkpoint."$(NOCOLOR)
	@read -p"Press ENTER to continue the steps or Ctrl-C to cancel" variable_nps; echo;
	@if [ -e ${TF_MODEL_DIRNAME}research/object_detection/export_food_graph ] ;\
	then \
		echo "Exported graph exists. Skipping exporting graph..." ;\
	else \
		echo "Models directory for TF Object Detection was not found" ;\
		echo "Please downloading training files using: make download_train_files" ;\
                cd ${TF_MODEL_DIRNAME}research ;\
                ./bin/protoc object_detection/protos/*.proto --python_out=. ;\
                export PYTHONPATH=$PYTHONPATH:`pwd`/object_detection ;\
                export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim ;\
                cd object_detection ;\
                mkdir export_food_graph ;\
                python3 export_inference_graph.py \
                --input_type=image_tensor \
                --pipeline_config_path=training_food/ssd_inception_v2_coco.config \
                --trained_checkpoint_prefix=training_food/model.ckpt-0 \
                --output_directory=export_food_graph ;\
	fi
	@echo "Compiling model using model optimizer..."
	@if [ -e ${TF_MODEL_DIRNAME}research/object_detection/export_food_graph ] ;\
	then \
                if [ -z "$(INTEL_OPENVINO_DIR)" ] ; \
                then \
                        echo "Please initiate the OpenVINO environment before optimizing TF graph."; \
                else \
                        echo "OpenVINO environment set. Continuing with optimizing TF graph..." ;\
			cd ${TF_MODEL_DIRNAME}research/object_detection/export_food_graph ;\
			mkdir ssd_inception_v2_mo_fp32_train ;\
			mkidr ssd_inception_v2_mo_fp16_train ;\
			${NCCOMPILE} \
			--data_type=FP32 --reverse_input_channels \
			--input_model frozen_inference_graph.pb \
			--tensorflow_object_detection_api_pipeline_config ../training_food/ssd_inception_v2_coco.config \
			--tensorflow_use_custom_operations_config ../../../../ssd_v2_support_inception_train.json \
			--output_dir ssd_inception_v2_mo_fp32_train ;\
			${NCCOMPILE} \
			--data_type=FP16 --reverse_input_channels \
			--input_model frozen_inference_graph.pb \
			--tensorflow_object_detection_api_pipeline_config ../training_food/ssd_inception_v2_coco.config \
			--tensorflow_use_custom_operations_config ../../../../ssd_v2_support_inception_train.json \
			--output_dir ssd_inception_v2_mo_fp16_train ;\
			mv ssd_inception_v2_mo_fp16_train ../../../../ssd_inception_v2_mo_fp16_train ;\
			mv ssd_inception_v2_mo_fp32_train ../../../../ssd_inception_v2_mo_fp32_train ;\
			echo "Successfully exported and optimized training graphs!" ;\
                fi ; \
	else \
		echo "No exported graph was made, so graph optimization could not be completed." ;\
	fi


.PHONY: train
train: train_info download_train_files set_up_training_api copy_training_files train_model


.PHONY: train_info
train_info:
	@echo $(YELLOW) '\n'${NETWORK_NAME}": Information disclosure before training model." $(NOCOLOR);
	@echo "The following steps will be taken in order to train your own model that detects produce:"
	@echo "  - Download training files"
	@echo "  - Set up and locally install training environment and API"
	@echo "  - Copy training files into desired location for training"
	@echo "  - Train model using Tensorflow Object Detection API"
	@echo ""
	@echo "Throughout this whole process, files will be downloaded and repositories will be cloned."
	@echo $(YELLOW)"Please pay attention to the prompts before proceeding as installation may error"
	@echo "if not followed properly."$(NOCOLOR)
	@read -p"Press ENTER to continue the steps above (this might take a long time) or Ctrl-C to cancel" variable_nps; echo;


.PHONY: train_model
train_model:
	@echo $(YELLOW) '\n'${NETWORK_NAME}": Training model..." $(NOCOLOR);
	@echo "  - Trained network  will be saved in:";
	@echo "      training_food";
	@echo ""
	@echo $(YELLOW)"URGENT: READ THE FOLLOWING INSTRUCTIONS BEFORE TRAINING:" ;
	@echo "In order to have more verbose logging so you can verify that your model is" ;
	@echo "actually training, adding the following to line 57 of model_main.py in" ;
	@echo "models/research/object_detection: tf.logging.set_verbosity(tf.logging.INFO)." 
	@echo "A reference file will be made with the original file content: old_model_main.py" $(NOCOLOR);
	@echo ""
	@echo "NOTE: Training files are unfrozen. If would like to export a model,";
	@echo "enter 'make export_and_optimize_model' to generate an optimized model";
	@echo "from your Tensorflow model.";
	@echo ""
	@echo "NOTE: If your system is running out of memory, please decrease batch size to 1.";
	@echo "Work upwards to empirically test if your system can handle increased batch sizes.";
	@echo "To edit batch size, edit the ssd_inception_v2_coco.config file in models/research/training_food by"
	@echo "replacing the number on line 136 to any positive integer (ex. 1) after 'batch_size:'."
	@echo ""
	@read -p"Press ENTER to continue the steps above (this might take a long time) or Ctrl-C to cancel" variable_nps; echo;
	
	@if [ -e ${TF_MODEL_DIRNAME}/research/object_detection/old_model_main.py ] ;\
	then \
		echo "Changing model_main.py" ;\
		cd models/research/object_detection ;\
		rm -rf model_main.py ;\
		cp old_model_main.py model_main.py ;\
		sed -i -e '57itf.logging.set_verbosity(tf.logging.INFO)' model_main.py ;\
	else \
		echo "Changing model_main.py and adding a reference file" ;\
		cd models/research/object_detection ;\
		cp model_main.py old_model_main.py ;\
		sed -i -e '57itf.logging.set_verbosity(tf.logging.INFO)' model_main.py ;\
	fi
	@if [ -e ${TF_MODEL_DIRNAME} ] ; \
	then \
		echo "Directory: ${TF_MODEL_DIRNAME}research/object_detection exists, setting up environment..."; \
		cd models/research/ ;\
		./bin/protoc object_detection/protos/*.proto --python_out=. ;\
		export PYTHONPATH=$PYTHONPATH:`pwd`/object_detection ;\
		export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim ;\
		cd object_detection ;\
		echo "Environment is setup, commencing with training..." ;\
		python3 model_main.py \
    --pipeline_config_path=training_food/ssd_inception_v2_coco.config \
    --model_dir=training_food/ \
    --num_train_steps=20000 \
    --alsologtostderr ;\
	else \
		echo "Cannot locate the training files for TF Object Detection. Please download."; \
	fi ;

.PHONY: set_up_training_api
set_up_training_api:
	@echo $(YELLOW)"Setting up training environment..."
	@if [ -e cocoapi/PythonAPI ] ; \
	then \
		echo $(YELLOW)"URGENT: READ FOLLOWING BEFORE PROCEEDING:" ;\
		echo "Replacing 'python' with 'python3' in the Makefile within cocoapi/PythonAPI"$(NOCOLOR) ;\
		read -p"Press ENTER to proceed or CTRL+C to exit..." variable_nps; echo ;\
		echo $(NOCOLOR)"Installing COCO API..." ;\
		cd cocoapi/PythonAPI ;\
		sed -i 's/python/python3/g' Makefile ;\
		make ;\
		cp -r pycocotools ../../models/research/ ;\
	else \
		echo $(NOCOLOR)"Please downloading training files before set up."; \
	fi
	@if [ -e ${TF_MODEL_DIRNAME} ] ; \
	then \
		echo $(NOCOLOR)"Models repo exists, proceeding with setup."; \
		echo $(NOCOLOR)"Setting up protobuf."; \
		cd models/research/ ;\
		unzip protobuf.zip ;\
		./bin/protoc object_detection/protos/*.proto --python_out=. ;\
		export PYTHONPATH=$PYTHONPATH:`pwd`/object_detection ;\
		export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim ;\
	else \
		echo $(NOCOLOR)"Please downloading training files before set up."; \
	fi
	@echo $(YELLOW)"Testing training environment..."
	@if python3 -c 'import pkgutil; exit(not pkgutil.find_loader("tensorflow"))' ; \
	then \
		echo $(NOCOLOR)'Tensorflow found, proceeding with testing the training environment...' ; \
		cd models/research/ ;\
		./bin/protoc object_detection/protos/*.proto --python_out=. ;\
		export PYTHONPATH=$PYTHONPATH:`pwd`/object_detection ;\
		export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim ;\
		python3 object_detection/builders/model_builder_test.py ;\
	else \
		echo 'Tensorflow not found. Please install tensorflow (CPU version).' ; \
		echo 'If you would like to install Tensorflow-gpu, please analyze your' ; \
		echo 'your systems hardware and install the required drivers. Then, pip/pip3' ; \
		echo 'install tensorflow-gpu (might need special permissions).' ; \
	fi
	@echo ""

.PHONY: copy_training_files
copy_training_files:
	@echo $(YELLOW)"Moving training files..."
	@echo $(NOCOLOR)"Will potentially overwrite training files in models/research/object_detection"
	@echo "Please check if there are training files within models/research/object_detection."
	@read -p"Press ENTER to proceed or CTRL+C to exit..." variable_nps; echo;
	@echo $(NOCOLOR)"Checking if training files directory exists in current directory..."
	@if [ -e ${TRAIN_DIR} ] ;\
	then \
		echo $(NOCOLOR)"Training files exist, proceeding to copy them into object detection directory."; \
		cp -a ${TRAIN_DIR}/ssd_inception_v2_coco_2018_01_28 models/research/object_detection/. ; \
		mkdir models/research/object_detection/training_food ;\
		cp -a ${TRAIN_DIR}/. models/research/object_detection/training_food/. ;\
		echo "Moved training files into models/research/object_detection" ;\
	else \
		echo $(NOCOLOR)"Please download training files."; \
	fi
	@echo ""

.PHONY: download_train_files
download_train_files:
	@echo $(YELLOW)"Downloading train files..."
	@if [ -e ${TRAIN_CHKPT_DIR} ] ; \
	then \
		echo $(NOCOLOR)"Checkpoint files already exist, skipping network request."; \
	else \
		echo $(NOCOLOR)"Grabbing checkpoint files from Tensorflow..."; \
		${PRE_TRAINED_MODEL} ;\
		tar -xvzf ${TRAIN_CHKPT_DIR}.tar.gz ;\
		mv ssd_inception_v2_coco_2018_01_28 ${TRAIN_CHKPT_DIR} ;\
	fi
	@if [ -e ${TEST_RECORD_FILE} ] ; \
	then \
		echo $(NOCOLOR)"Test record file already exists, skipping network request."; \
	else \
		echo $(NOCOLOR)"Grabbing test record file..."; \
		${TEST_RECORD} ;\
	fi
	@if [ -e ${TRAIN_PBTXT_FILE} ] ; \
	then \
		echo $(NOCOLOR)"Pbtxt file already exists, skipping network request."; \
	else \
		echo $(NOCOLOR)"Grabbing pbtxt file..."; \
		${TRAIN_PBTXT} ;\
	fi
	@if [ -e ${SSD_CONFIG_FILE} ] ; \
	then \
		echo $(NOCOLOR)"SSD Inception v2 config already exists, skipping network request."; \
	else \
		echo $(NOCOLOR)"Grabbing SSD Inception v2 config file..."; \
		${SSD_CONFIG} ;\
	fi
	@if [ -e ${TF_MODEL_DIRNAME} ] ; \
	then \
		echo $(NOCOLOR)"Models repo already exists, skipping network request."; \
	else \
		echo $(NOCOLOR)"Cloning models repo..."; \
		${TF_MODEL_REPO} ;\
	fi
	@if [ -e ${PROTOC_FILE} ] ; \
	then \
		echo $(NOCOLOR)"Protoc file already exists, skipping network request."; \
	else \
		echo $(NOCOLOR)"Grabbing protoc file..."; \
		${PROTOC_DOWNLOAD} ;\
	fi
	@if [ -e ${ANNO_ZIP_FILE} ] ; \
	then \
		echo $(NOCOLOR)"Annotated zip file already exists, skipping network request."; \
	else \
		echo $(NOCOLOR)"Grabbing annotated zip file..."; \
		${ANNO_DOWNLOAD} ;\
		unzip ${ANNO_ZIP_FILE} -d train_files/ ;\
	fi
	@echo "Downloading larger files using gdown, checking for installation"
	pip3 install --user gdown
	@if [ -e ${TRAIN_RECORD_FILE} ] ; \
	then \
		echo $(NOCOLOR)"Train record already exists, skipping network request."; \
	else \
		echo $(NOCOLOR)"Grabbing train record file..."; \
		wget -O train_files/train.record 'https://downloadmirror.intel.com/29611/eng/food_detection_training_files.zip' ;\
	fi
	@if [ -e ${JPEG_ZIP_FILE} ] ; \
	then \
		echo $(NOCOLOR)"JPEG zip file already exists, skipping network request."; \
	else \
		echo $(NOCOLOR)"Grabbing JPEG zip file..."; \
		wget -O train_files/JPEGImages.zip 'https://downloadmirror.intel.com/29611/eng/food_detection_model_labels.zip' ;\
		unzip ${JPEG_ZIP_FILE} -d train_files/ ;\
	fi
	@if [ -e cocoapi/PythonAPI ] ;\
	then \
		echo $(NOCOLOR)"cocoapi repo already exists, skipping network request."; \
	else \
		echo "Cloning cocoapi repo for future installation..." ;\
		git clone https://github.com/cocodataset/cocoapi.git ;\
	fi
	@echo ""

.PHONY: uninstall-reqs
uninstall-reqs: 
	@echo $(YELLOW)"\nSSD Inception V2 Food: Uninstalling requirements..."$(NOCOLOR)
	@echo "Nothing to uninstall."

.PHONY: help
help:
	@echo "\nPossible make targets"
	@echo $(YELLOW)"  make help " $(NOCOLOR) "- Shows this message.";
	@echo $(YELLOW)"  make run or run_py " $(NOCOLOR) "- Runs the Python example.";
	@echo $(YELLOW)"  make all " $(NOCOLOR) "- Makes everything needed to run, but doesn't run the sample.";
	@echo $(YELLOW)"  make compile_model " $(NOCOLOR) "- Runs model compiler for the network.";
	@echo $(YELLOW)"  make data " $(NOCOLOR) "- Downloads required data.";
	@echo $(YELLOW)"  make deps " $(NOCOLOR) "- Makes dependencies for project, prepares model etc.";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make clean " $(NOCOLOR) "- Removes all the temporary files that are created by the Makefile.";
	@echo $(YELLOW)"  make train " $(NOCOLOR) "- (TBD) Train the mnist model.";

clean: clean
	@echo $(YELLOW)"\nSSD Inception V2 Food: Cleaning up files..."$(NOCOLOR)
	rm -rf ${TEST_IMG_DIR}
	rm -rf ${TRAIN_DIR}
	rm -rf models
	rm -rf cocoapi
	rm -rf ssd_inception_v2_mo_fp16_train
	rm -rf ssd_inception_v2_mo_fp32_train
	rm -f frozen_inference_graph.pb
	rm -f frozen_inference_graph.labels
	rm -f food_detection_model_labels.zip
	rm -f frozen_inference_graph.bin
	rm -f frozen_inference_graph.xml
	rm -f frozen_inference_graph.mapping
	rm -f ssd_inception_v2_coco.config