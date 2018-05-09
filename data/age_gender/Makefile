
.PHONY: all

all:
	@test -f age_gender_mean.npy || wget -P . -N https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/mean.binaryproto
	@python3 convert_mean.py

.PHONY: clean
clean:
	@rm -f age_gender_mean.npy
	@rm -f mean.binaryproto
