
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:/opt/movidius/mvnc/python:$(PYTHONPATH)
endif

.PHONY: help
help:
	@echo "Possible Make targets"
	@echo "  make help - shows this message"
	@echo "  make clean - Removes all temp files from all directories"

TOPTARGETS := clean

SUBDIRS := ilsvrc12 age_gender

$(TOPTARGETS): $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

.PHONY: $(TOPTARGETS) $(SUBDIRS)

all compile profile check run:

