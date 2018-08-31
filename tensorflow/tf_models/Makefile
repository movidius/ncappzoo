
PROJECT_DIR = $(CURDIR)

YELLOW='\033[1;33m'
NOCOLOR='\033[0m'

.PHONY: all
all: check_model

GIT_REPO = https://github.com/tensorflow/models.git

check_model:
ifndef TF_MODELS_PATH
	@echo "TF_MODELS_PATH not set, will use project directory";
	$(eval TF_MODELS_PATH=${PROJECT_DIR}/models)
	@echo "TF_MODELS_PATH is now: ${TF_MODELS_PATH}"
	@if [ -d ${TF_MODELS_PATH} ] ; \
	then \
		echo "Skipping clone, directory already exists: ${TF_MODELS_PATH}" ; \
	else \
		echo "Cloning ${GIT_REPO}"; \
		git clone ${GIT_REPO}; \
		cd models; \
		git checkout 08af77755a1cfbc05835d77b3261c92b314b4463; \
		cd ..; \
	fi
else
	@echo "TF_MODELS_PATH is ${TF_MODELS_PATH}" ;
endif


.PHONY: help
help:
	@echo $(YELLOW)"\nPossible make targets: "$(NOCOLOR);
	@echo "  make help - Shows this message.";
	@echo "  make clean - Deletes everything created in this directory";
	@echo "  make compile - Compiles all";
	@echo "  make all - Downloads tensorflow models repo from here:"
	@echo "             ${GIT_REPO}" 

.PHONY: clean
clean:
	@echo $(YELLOW)"\nMaking clean..."$(NOCOLOR);
	rm -rf models

compile: all



