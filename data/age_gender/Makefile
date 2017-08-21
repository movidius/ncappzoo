
.PHONY: all

all:
	@test -f age_gender_mean.npy || wget -P . -N https://dl.dropboxusercontent.com/u/38822310/mean.binaryproto
	@python3 convert_mean.py

.PHONY: clean
clean:
	@rm -f age_gender_mean.npy
	@rm -f mean.binaryproto
