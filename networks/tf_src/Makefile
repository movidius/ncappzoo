
PROJECT_DIR = $(CURDIR)

YELLOW='\033[1;33m'
NOCOLOR='\033[0m'

.PHONY: all
all: check_tf

GIT_REPO = https://github.com/tensorflow/tensorflow.git

check_tf:
ifndef TF_SRC_PATH
	@echo "TF_SRC_PATH not set, will use project directory";
	$(eval TF_SRC_PATH=${PROJECT_DIR}/tensorflow)
	@echo "TF_SRC_PATH is now: ${TF_SRC_PATH}"
	@if [ -d ${TF_SRC_PATH} ] ; \
	then \
		echo "skipping clone, directory already exists: ${TF_SRC_PATH}" ; \
	else \
		echo "cloning ${GIT_REPO}"; \
		git clone -b r1.7 ${GIT_REPO}; \
	fi
else
	@echo "TF_SRC_PATH is ${TF_SRC_PATH}" ;
endif


.PHONY: help
help:
	@echo $(YELLOW)"\nPossible make targets: "$(NOCOLOR);
	@echo "  make help - Shows this message.";
	@echo "  make compile - Compiles all.";
	@echo "  make clean - Deletes everything created in this directory";
	@echo "  make all - Downloads tensorflow models repo from here:"
	@echo "             ${GIT_REPO}" 

.PHONY: clean
clean:
	@echo $(YELLOW)"\nMaking clean..."$(NOCOLOR);
	rm -rf tensorflow

compile: all

