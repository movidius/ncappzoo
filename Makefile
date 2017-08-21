
.PHONY: validate
validate:
	@(cd Caffe/GoogLeNet; make)
	@(cd Caffe/AlexNet; make)
	@(cd Caffe/SqueezeNet; make)
	@(cd Caffe/AgeNet; make)
	@(cd Caffe/GenderNet; make)

.PHONY: clean
clean:
	@(cd Caffe/GoogLeNet; make clean)
	@(cd Caffe/AlexNet; make clean)
	@(cd Caffe/SqueezeNet; make clean)
	@(cd Caffe/AgeNet; make clean)
	@(cd Caffe/GenderNet; make clean)
	@(cd data/age_gender; make clean)
	@(cd data/ilsvrc12; make clean)
