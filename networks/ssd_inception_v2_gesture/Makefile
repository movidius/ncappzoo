NCCOMPILE = mo.py
CPU_EXTENSION = $(INTEL_OPENVINO_DIR)/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so

YELLOW='\033[1;33m'
NOCOLOR='\033[0m'
RED = '\033[1;31m'

GRAPH_FILENAME_BASE = ssd_inception_v2_gesture
GRAPH_DIR_FP32 = ssd_inception_v2_mo_fp32
GRAPH_DIR_FP16 = ssd_inception_v2_mo_fp16
GRAPH_FILENAME_BASE_IN_DIR_FP32 = ${GRAPH_DIR_FP32}/frozen_inference_graph.xml
GRAPH_FILENAME_BASE_IN_DIR_FP16 = ${GRAPH_DIR_FP16}/frozen_inference_graph.xml
MO_LABELS = frozen_inference_graph.labels
MO_LABELS_IN_DIR_FP32 = ${GRAPH_DIR_FP32}/frozen_inference_graph.labels
MO_LABELS_IN_DIR_FP16 = ssd_inception_v2_mo_fp16/frozen_inference_graph.labels
MO_INPUT_GRAPH_ARG = --input_model tensorflow_model/frozen_inference_graph.pb
MO_INPUT_PIPELINE = --tensorflow_object_detection_api_pipeline_config tensorflow_model/pipeline.config
MO_INPUT_CUSTOM_OPS = --tensorflow_use_custom_operations_config ssd_v2_support_inception.json
MO_OUTPUT_DIR_FP32 = --output_dir ${GRAPH_DIR_FP32}
MO_OUTPUT_DIR_FP16 = --output_dir ${GRAPH_DIR_FP16}

MODEL_DIR = tensorflow_model
MODEL_PIPELINE_CONFIG = ${MODEL_DIR}/pipeline.config
MODEL_PB = ${MODEL_DIR}/frozen_inference_graph.pb
MODEL_INFERENCE_PIPELINE_CONFIG = pipeline.config
MODEL_INFERENCE_PB = frozen_inference_graph.pb

GET_MODEL_PIPELINE = wget -c --no-cache -P ./tensorflow_model https://raw.githubusercontent.com/fcr3/gesture_detection/master/tensorflow_model/${MODEL_INFERENCE_PIPELINE_CONFIG}
GET_MO_MODEL_FP32_LABELS = wget -c --no-cache -P ./ssd_inception_v2_mo_fp32 https://raw.githubusercontent.com/fcr3/gesture_detection/master/model_optimized_fp32/${MO_LABELS}
GET_MO_MODEL_FP16_LABELS = wget -c --no-cache -P ./ssd_inception_v2_mo_fp16 https://raw.githubusercontent.com/fcr3/gesture_detection/master/model_optimized_fp16/${MO_LABELS}
GET_MODEL_PB = wget http://dp2f0iayzbweh.cloudfront.net/ncappzoo/gesture_detection/model/frozen_inference_graph.pb -O ${MODEL_DIR}/${MODEL_INFERENCE_PB}
# if above doesn't work: python3 download_weights.py

RUN_PY_RELATIVE_DIR = ssd_inception_v2_gesture.py

TRAIN_DIR = train_files
TRAIN_CHKPT_DIR = ${TRAIN_DIR}/ssd_inception_v2_coco_2018_01_28
PRE_TRAINED_MODEL = wget -c --no-cache -P ./${TRAIN_DIR} http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz

TRAIN_RECORD_FILE = ./${TRAIN_DIR}/train.record
TRAIN_RECORD = wget http://dp2f0iayzbweh.cloudfront.net/ncappzoo/gesture_detection/training/train.record -O ${TRAIN_RECORD_FILE}
# TRAIN_RECORD_LINK = 'https://drive.google.com/uc?id=13uM9aoxrUvPjZTIkXv33av_Dx8CYhg7S'

TEST_RECORD_FILE = ./${TRAIN_DIR}/test.record
TEST_RECORD = wget http://dp2f0iayzbweh.cloudfront.net/ncappzoo/gesture_detection/training/test.record -O ${TEST_RECORD_FILE}

TRAIN_PBTXT_FILE = ./${TRAIN_DIR}/object-detection.pbtxt
TRAIN_PBTXT = wget http://dp2f0iayzbweh.cloudfront.net/ncappzoo/gesture_detection/training/object-detection.pbtxt -O ${TRAIN_PBTXT_FILE}

SSD_CONFIG_FILE = ./${TRAIN_DIR}/ssd_inception_v2_coco.config
SSD_CONFIG = wget http://dp2f0iayzbweh.cloudfront.net/ncappzoo/gesture_detection/training/ssd_inception_v2_coco.config -O ${SSD_CONFIG_FILE}

TRAIN_PIPE_CONFIG_FILE = ./${TRAIN_DIR}/pipeline.config
TRAIN_PIPE_CONFIG = wget http://dp2f0iayzbweh.cloudfront.net/ncappzoo/gesture_detection/training/pipeline.config -O ${TRAIN_PIPE_CONFIG_FILE}

JPEG_ZIP_FILE = ./${TRAIN_DIR}/JPEGImages.zip
JPEG_DOWNLOAD = wget http://dp2f0iayzbweh.cloudfront.net/ncappzoo/gesture_detection/training/JPEGImages-20200116T001612Z-001.zip -O ${JPEG_ZIP_FILE}
# JPEG_FOLDER_LINK = 'https://drive.google.com/uc?id=1J5p63WjnkcWAyCQVbaW6qshsUBfX7aZ2'

ANNO_ZIP_FILE = ./${TRAIN_DIR}/Annotations.zip
ANNO_DOWNLOAD = wget http://dp2f0iayzbweh.cloudfront.net/ncappzoo/gesture_detection/training/Annotations-20200115T233047Z-001.zip -O ${ANNO_ZIP_FILE}
# ANNO_FOLDER_LINK = 'https://drive.google.com/uc?id=17ZmWWQFV5ljMWTX1HP-dZtD25xKXVCYU'

NETWORK_NAME = SSD Inception v2: Gesture Model
TF_MODEL_DIRNAME = models/
TF_MODEL_REPO = git clone https://github.com/tensorflow/models.git
PROTOC_FILE = models/research/protobuf.zip
PROTOC_DOWNLOAD = wget -O ${PROTOC_FILE} https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip

.PHONY: all
all: deps data compile_model

.PHONY: deps
deps: get_model
	@echo $(YELLOW)"\nSSD Inception V2 Gesture: Making dependencies... No extra dependencies needed."$(NOCOLOR)


.PHONY: data
data:
	@echo $(YELLOW)"\nSSD Inception V2 Gesture: Downloading required data... No data needed."$(NOCOLOR)


.PHONY: get_model
get_model:
	@echo $(YELLOW)"\nSSD Inception V2 Gesture: Downloading model files..."$(NOCOLOR)
	@if [ -e ${MODEL_PIPELINE_CONFIG} ] ; \
	then \
		echo "model file ${MODEL_PIPELINE_CONFIG} already exists, skipping download"; \
	else \
		echo "Downloading ${MODEL_INFERENCE_PIPELINE_CONFIG} file"; \
		${GET_MODEL_PIPELINE}; \
		if [ -e ${MODEL_PIPELINE_CONFIG} ] ; \
		then \
			echo "download ${MODEL_INFERENCE_PIPELINE_CONFIG} done. Downloaded to /tensorflow_model."; \
		else \
			echo "***\nError - Could not download ${MODEL_INFERENCE_PIPELINE_CONFIG}. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi
	@if [ -e ${MODEL_PB} ] ; \
	then \
		echo "model file ${MODEL_PB} already exists, skipping download"; \
	else \
		echo "Downloading ${MODEL_INFERENCE_PB} file" ; \
		${GET_MODEL_PB} ; \
		if [ -e ${MODEL_PB} ] ; \
		then \
			echo "download ${MODEL_INFERENCE_PB} done. Downloaded to /tensorflow_model."; \
		else \
			echo "***\nError - Could not download ${MODEL_INFERENCE_PB}. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi

.PHONY: compile_from_training
compile_from_training:
	@echo $(YELLOW) '\n'${NETWORK_NAME}": Information disclosure before compiling model." $(NOCOLOR);
	@echo "The following steps will be taken in order to compile your own model that detects produce:"
	@echo "  - Freeze trained model"
	@echo "  - Run frozen model through model optimizer"
	@echo ""
	@echo $(YELLOW)"Before proceeding, edit the compile_from_training script in the Makefile."
	@echo "Specifically, edit the trained_checkpoin_prefix to reflect your true (latest) model checkpoint."$(NOCOLOR)
	@read -p"Press ENTER to continue the steps or Ctrl-C to cancel" variable_nps; echo;
	@if [ -e ${TF_MODEL_DIRNAME}research/object_detection/export_gesture_graph ] ;\
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
    mkdir export_gesture_graph ;\
    python3 export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=training_gesture/ssd_inception_v2_coco.config \
    --trained_checkpoint_prefix=training_gesture/model.ckpt-0 \
    --output_directory=export_gesture_graph ;\
	fi
	@echo "Compiling model using model optimizer..."
	@if [ -e ${TF_MODEL_DIRNAME}research/object_detection/export_gesture_graph ] ;\
	then \
    if [ -z "$(INTEL_OPENVINO_DIR)" ] ; \
    then \
            echo "Please initiate the OpenVINO environment before optimizing TF graph."; \
    else \
            echo "OpenVINO environment set. Continuing with optimizing TF graph..." ;\
						cd ${TF_MODEL_DIRNAME}research/object_detection/export_gesture_graph ;\
						mkdir ssd_inception_v2_mo_fp32_train ;\
						mkidr ssd_inception_v2_mo_fp16_train ;\
						${NCCOMPILE} \
						--data_type=FP32 --reverse_input_channels \
						--input_model frozen_inference_graph.pb \
						--tensorflow_object_detection_api_pipeline_config ../training_gesture/ssd_inception_v2_coco.config \
						--tensorflow_use_custom_operations_config ../../../../ssd_v2_support_inception_train.json \
						--output_dir ssd_inception_v2_mo_fp32_train ;\
						${NCCOMPILE} \
						--data_type=FP16 --reverse_input_channels \
						--input_model frozen_inference_graph.pb \
						--tensorflow_object_detection_api_pipeline_config ../training_gesture/ssd_inception_v2_coco.config \
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
	@echo "Downloading larger files using gdown, checking for installation"
	@if [ -e ${ANNO_ZIP_FILE} ] ; \
	then \
		echo $(NOCOLOR)"Annotated zip file already exists, skipping network request."; \
	else \
		echo $(NOCOLOR)"Grabbing annotated zip file..."; \
		${ANNO_DOWNLOAD} ; \ # gdown ${ANNO_FOLDER_LINK} -O ${ANNO_ZIP_FILE} ;\ ;\
		unzip ${ANNO_ZIP_FILE} -d train_files/ ;\
	fi
	@if [ -e ${TRAIN_RECORD_FILE} ] ; \
	then \
		echo $(NOCOLOR)"Train record already exists, skipping network request."; \
	else \
		echo $(NOCOLOR)"Grabbing train record file..."; \
		${TRAIN_RECORD} ; \ # gdown ${TRAIN_RECORD_LINK} -O train_files/train.recor ;\
	fi
	@if [ -e ${JPEG_ZIP_FILE} ] ; \
	then \
		echo $(NOCOLOR)"JPEG zip file already exists, skipping network request."; \
	else \
		echo $(NOCOLOR)"Grabbing JPEG zip file..."; \
		${JPEG_DOWNLOAD} ; \ # gdown ${JPEG_FOLDER_LINK} -O train_files/JPEGImages.zi ;\
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
		mkdir models/research/object_detection/training_gesture ;\
		cp -a ${TRAIN_DIR}/. models/research/object_detection/training_gesture/. ;\
		echo "Moved training files into models/research/object_detection" ;\
	else \
		echo $(NOCOLOR)"Please download training files."; \
	fi
	@echo ""


.PHONY: train_model
train_model:
	@echo $(YELLOW) '\n'${NETWORK_NAME}": Training model..." $(NOCOLOR);
	@echo "  - Trained network  will be saved in:";
	@echo "      training_gesture";
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
	@echo "To edit batch size, edit the ssd_inception_v2_coco.config file in models/research/training_gesture by"
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
    --pipeline_config_path=training_gesture/ssd_inception_v2_coco.config \
    --model_dir=training_gesture/ \
    --num_train_steps=20000 \
    --alsologtostderr ;\
	else \
		echo "Cannot locate the training files for TF Object Detection. Please download."; \
	fi ;


.PHONY: compile_model
compile_model: get_model
	@echo $(YELLOW)"\nSSD Inception V2 Gesture: Compiling model to IR..."$(NOCOLOR)
	@if [ -e ${GRAPH_FILENAME_BASE_IN_DIR_FP32} ] ; \
	then \
		echo "compiled FP32 model file already exists, skipping compile."; \
	else \
		echo "Compiling FP32 model..."; \
		mkdir ssd_inception_v2_mo_fp32 ; \
		${NCCOMPILE} --data_type=FP32 --reverse_input_channels ${MO_INPUT_GRAPH_ARG} ${MO_INPUT_PIPELINE} ${MO_INPUT_CUSTOM_OPS} ${MO_OUTPUT_DIR_FP32} || (echo $(RED)"Make sure to set the OpenVINO environment variables using the "$(YELLOW)"setupvars.sh"$(RED)" script found in <your OpenVINO install location>/bin/ folder."$(NOCOLOR);  exit 1;); \
	fi
	@if [ -e ${GRAPH_FILENAME_BASE_IN_DIR_FP16} ] ; \
	then \
		echo "compiled FP16 model file already exists, skipping compile."; \
	else \
		echo "Compiling FP16 model..."; \
		mkdir ssd_inception_v2_mo_fp16 ; \
		${NCCOMPILE} --data_type=FP16 --reverse_input_channels ${MO_INPUT_GRAPH_ARG} ${MO_INPUT_PIPELINE} ${MO_INPUT_CUSTOM_OPS} ${MO_OUTPUT_DIR_FP16} || (echo $(RED)"Make sure to set the OpenVINO environment variables using the "$(YELLOW)"setupvars.sh"$(RED)" script found in <your OpenVINO install location>/bin/ folder."$(NOCOLOR);  exit 1;); \
	fi
	@if [ -e ${MO_LABELS_IN_DIR_FP32} ] ; \
	then \
		echo "compiled FP32 model labels already exist, skipping network request."; \
	else \
		echo "Grabbing labels for FP32 model..."; \
		${GET_MO_MODEL_FP32_LABELS}; \
	fi
	@if [ -e ${MO_LABELS_IN_DIR_FP16} ] ; \
	then \
		echo "compiled FP16 model labels already exist, skipping network request."; \
	else \
		echo "Grabbing labels for FP16 model..."; \
		${GET_MO_MODEL_FP16_LABELS} ;\
	fi

.PHONY: install-reqs
install-reqs:
	@echo $(YELLOW)"\nSSD Inception V2 Gesture: Checking application requirements...\n"$(NOCOLOR)
	@echo "Nothing to install..."


.PHONY: run_FP16
run_FP16: install-reqs deps data compile_model
	@echo $(YELLOW) "\nSSD Inception V2: Running Python sample..." $(NOCOLOR)
	python3 $(RUN_PY_RELATIVE_DIR) -i 'cam' -m ${GRAPH_FILENAME_BASE_IN_DIR_FP16} --labels ${MO_LABELS_IN_DIR_FP16} -d MYRIAD


.PHONY: run_FP32
run_FP32: install-reqs deps data compile_model
	@echo $(YELLOW) "\nSSD Inception V2: Running Python sample..." $(NOCOLOR)
	python3 $(RUN_PY_RELATIVE_DIR) -i 'cam' -m ${GRAPH_FILENAME_BASE_IN_DIR_FP32} -l ${CPU_EXTENSION} --labels ${MO_LABELS_IN_DIR_FP32}


.PHONY: run
run: run_py


.PHONY: run_py
run_py: install-reqs deps data compile_model run_FP16


.PHONY: uninstall-reqs
uninstall-reqs:
	@echo $(YELLOW)"\nSSD Inception V2 Gesture: Uninstalling requirements..."$(NOCOLOR)
	@echo "\nNo requirements to install..."


.PHONY: help
help:
	@echo "\nPossible make targets"
	@echo $(YELLOW)"  make help " $(NOCOLOR) "- Shows this message.";
	@echo $(YELLOW)"  make run or run_py " $(NOCOLOR) "- Runs the Python example.";
	@echo $(YELLOW)"  make run_FP16 "  $(NOCOLOR) "- Runs the Python example with FP16 model.";
	@echo $(YELLOW)"  make run_FP32 " $(NOCOLOR) "- Runs the Python example with FP32 model.";
	@echo $(YELLOW)"  make all " $(NOCOLOR) "- Makes everything needed to run, but doesn't run the sample.";
	@echo $(YELLOW)"  make compile_model " $(NOCOLOR) "- Runs model compiler for the network.";
	@echo $(YELLOW)"  make get_model " $(NOCOLOR) "- Downloads the model files.";
	@echo $(YELLOW)"  make data " $(NOCOLOR) "- Downloads required data.";
	@echo $(YELLOW)"  make deps " $(NOCOLOR) "- Makes dependencies for project, prepares model etc.";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make clean " $(NOCOLOR) "- Removes all the temporary files that are created by the Makefile.";
	@echo $(YELLOW)"  make train " $(NOCOLOR) "- Train a SSD Inception V2  model.";

clean: clean
	@echo $(YELLOW)"\nSSD Inception V2 Gesture: Cleaning up files..."$(NOCOLOR)
	rm -rf ${GRAPH_DIR_FP32}
	rm -rf ${GRAPH_DIR_FP16}
	rm -rf ${MODEL_DIR}
	rm -rf train_files
	rm -rf models
	rm -rf cocoapi
	rm -rf ssd_inception_v2_mo_fp16_train
	rm -rf ssd_inception_v2_mo_fp32_train
