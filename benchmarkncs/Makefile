
.PHONY: all
all: googlenet alexnet squeezenet agenet gendernet

.PHONY: googlenet
googlenet:
	@(cd ../Caffe/GoogLeNet; make compile)
	@./benchmarkncs.py ../Caffe/GoogLeNet ../data/images 224 224

.PHONY: alexnet
alexnet:
	@(cd ../Caffe/AlexNet; make compile)
	@./benchmarkncs.py ../Caffe/AlexNet ../data/images 227 227

.PHONY: squeezenet
squeezenet:
	@(cd ../Caffe/SqueezeNet; make compile)
	@./benchmarkncs.py ../Caffe/SqueezeNet ../data/images 227 227

.PHONY: agenet
agenet:
	@(cd ../Caffe/AgeNet; make compile)
	@./benchmarkncs.py ../Caffe/AgeNet ../data/images 227 227

.PHONY: gendernet
gendernet:
	@(cd ../Caffe/GenderNet; make compile)
	@./benchmarkncs.py ../Caffe/GenderNet ../data/images 227 227

.PHONY: clean
clean:
