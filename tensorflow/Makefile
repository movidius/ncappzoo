
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:/opt/movidius/mvnc/python:$(PYTHONPATH)
endif


SUBDIRS := $(wildcard */.)
TOPTARGETS := all clean check compile profile run

$(TOPTARGETS): $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

.PHONY: $(TOPTARGETS) $(SUBDIRS)

.PHONY: help
help:
	@echo "Possible Make targets"
	@echo "  make help - shows this message"
	@echo "  make all - Makes all targets"
	@echo "  make clean - Removes all temp files from all directories"
	@echo "  make check - Runs check on all caffe/tensorflow models"
	@echo "  make profile - Runs profile on all caffe/tensorflow models"
	@echo "  make compile - Runs compile on all caffe/tensorflow models"
	@echo "  make run - Runs all caffe/tensorflow/apps"
