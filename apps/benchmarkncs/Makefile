
.PHONY: all
all: googlenet alexnet squeezenet agenet gendernet

.PHONY: googlenet
googlenet:
	@./benchmarkncs.py ../../caffe/GoogLeNet ../../data/images 224 224

.PHONY: alexnet
alexnet:
	@./benchmarkncs.py ../../caffe/AlexNet ../../data/images 227 227

.PHONY: squeezenet
squeezenet:
	@./benchmarkncs.py ../../caffe/SqueezeNet ../../data/images 227 227

.PHONY: agenet
agenet:
	@./benchmarkncs.py ../../caffe/AgeNet ../../data/images 227 227

.PHONY: gendernet
gendernet:
	@./benchmarkncs.py ../../caffe/GenderNet ../../data/images 227 227

.PHONY: inception_v1
inception_v1:
	@./benchmarkncs.py ../../tensorflow/inception_v1 ../../data/images 227 227

.PHONY: inception_v3
inception_v3:
	@./benchmarkncs.py ../../tensorflow/inception_v3 ../../data/images 299 299

.PHONY: clean
clean:

compile profile check:
