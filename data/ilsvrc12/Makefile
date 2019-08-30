python_version_full := $(wordlist 2,4,$(subst ., ,$(shell python3 --version 2>&1)))
python_version_minor := $(word 2,${python_version_full})

GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'

ILSVRC_SH = get_ilsvrc_aux.sh
GET_ILSVRC_AUX = wget https://raw.githubusercontent.com/BVLC/caffe/master/data/ilsvrc12/get_ilsvrc_aux.sh
.PHONY: all

all: 
	@echo $(YELLOW)"\nMaking all for ilsvrc data... "$(NOCOLOR);
	@test -f ${ILSVRC_SH} || (${GET_ILSVRC_AUX};chmod +x ${ILSVRC_SH};./${ILSVRC_SH});
	@test -f ilsvrc_2012_mean.npy || wget https://raw.githubusercontent.com/BVLC/caffe/master/python/caffe/imagenet/ilsvrc_2012_mean.npy; 
	@cut -c11- synset_words.txt >> synset_labels.txt; \
	
.PHONY: clean
clean:
	@echo $(YELLOW)"\nMaking clean for ilsvrc data..."$(NOCOLOR);
	@rm -f ${ILSVRC_SH}
	@rm -f test.txt
	@rm -f synsets.txt
	@rm -f val.txt
	@rm -f train.txt
	@rm -f synset_words.txt
	@rm -f det_synset_words.txt
	@rm -f imagenet.bet.pickle
	@rm -f imagenet_mean.binaryproto
	@rm -f ._*
	@rm -f caffe_ilsvrc12.tar.gz
	@rm -f ilsvrc_2012_mean.npy
	@rm -f synset_labels.txt
	
