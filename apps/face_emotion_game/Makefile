GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'

# filenames for the graph files that we'll copy to this directory.
FACE_RELATIVE_DIR = ../../networks/face_detection_retail_0004
FACE_FILE_NAME_BASE = face-detection-retail-0004

EMOTION_RELATIVE_DIR = ../../networks/emotions_recognition_retail_0003
EMOTION_FILE_NAME_BASE = emotions-recognition-retail-0003


# name of the directory and the base name of the main python file (minus the .py extension)
APP_NAME = face_emotion_game

.PHONY: all
all: deps data compile_cpp


.PHONY: data
data:
	@echo $(YELLOW)'\n'${APP_NAME}": No data required."$(NOCOLOR)


.PHONY: deps
deps: get_ir
	@echo $(YELLOW)'\n'${APP_NAME}": Making dependencies..."$(NOCOLOR)



.PHONY: default_model
default_model: get_ir
	@echo $(YELLOW)'\n'${APP_NAME}": Making default models..."$(NOCOLOR)


.PHONY: get_ir
get_ir:
	@echo $(YELLOW)'\n'${APP_NAME}": Downloading IRs..."$(NOCOLOR);
	@if [ -e ${EMOTION_FILE_NAME_BASE}.xml ] && [ -e ${EMOTION_FILE_NAME_BASE}.bin ] ;\
	then \
		echo " - Emotion Recognition IRs already exist in the project folder.";\
	else \
		echo " - Emotion Recognition IR files do not exist in project directory.";\
		echo " - Making Emotion Recognition IRs...";\
		(cd ${EMOTION_RELATIVE_DIR}; make get_ir;);\
		echo " - Copying Emotion Recognition IR files to current folder..." ;\
		cp ${EMOTION_RELATIVE_DIR}/${EMOTION_FILE_NAME_BASE}.xml src/data/emotions-recognition/ ;\
		cp ${EMOTION_RELATIVE_DIR}/${EMOTION_FILE_NAME_BASE}.bin src/data/emotions-recognition/ ;\
	fi;\

	@if [ -e ${FACE_GEN_MODEL_FILE_NAME_BASE}.xml ] && [ -e ${FACE_GEN_MODEL_FILE_NAME_BASE}.bin ] ;\
	then \
		echo " - Face detection model IR already exist in the project folder.";\
	else \
		echo " - Face detection model IR files do not exist in project directory.";\
		echo " - Making Face detection IRs...";\
		(cd ${FACE_RELATIVE_DIR}; make get_ir;);\
		echo " - Copying Face IR files to current folder..." ;\
		cp ${FACE_RELATIVE_DIR}/${FACE_FILE_NAME_BASE}.xml src/data/face-detection/ ;\
		cp ${FACE_RELATIVE_DIR}/${FACE_FILE_NAME_BASE}.bin src/data/face-detection/ ;\
	fi


.PHONY:	compile_cpp
compile_cpp:
	@echo $(YELLOW)'\n'${APP_NAME}": Building the sample..."$(NOCOLOR)
	@if [ -z "$(INTEL_OPENVINO_DIR)" ] ; \
	then \
		echo "Please initiate the Intel OpenVINO environment by going to the installation directory for openvino and running the setupvars.sh file in the bin folder." ; \
		exit 1 ; \
	else \
		echo "Intel OpenVINO environment is already set!" ; \
	fi
	(mkdir -p build; cd build; cmake ..; make;)


.PHONY: run
run: run_cpp


.PHONY: run_cpp
run_cpp: deps data compile_cpp
	@echo $(YELLOW)'\n'${APP_NAME}": Running the sample..."$(NOCOLOR)
	(cd build; ./${APP_NAME}; cd ..;)


.PHONY: install-reqs
install-reqs:
	@echo $(YELLOW)"\n"$(APP_NAME)": Checking installation requirements..."$(NOCOLOR)
	@echo "No requirements needed."


.PHONY: uninstall-reqs
uninstall-reqs:
	@echo $(YELLOW)'\n'${APP_NAME}": Uninstalling requirements..."$(NOCOLOR)
	@echo "Nothing to uninstall."


.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run or run_cpp"$(NOCOLOR)"- runs the application";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes everything needed to run but doesn't run";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- downloads data as needed";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- makes/prepares dependencies";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make default_model "$(NOCOLOR)"- compiles a default model to use when running";
	@echo $(YELLOW)"  make get_ir "$(NOCOLOR)"- gets the age-gender/face IR files from the model zoo";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content"
	@echo ""


clean:
	@echo $(YELLOW)'\n'${APP_NAME}": Cleaning up files..."$(NOCOLOR);
	rm -rf build
	rm -f src/emotions-recognition/${EMOTION_RELATIVE_DIR}.xml;
	rm -f src/emotions-recognition/${EMOTION_RELATIVE_DIR}.bin;
	rm -f src/face-detection/${FACE_FILE_NAME_BASE}.xml;
	rm -f src/face-detection/${FACE_FILE_NAME_BASE}.bin;

