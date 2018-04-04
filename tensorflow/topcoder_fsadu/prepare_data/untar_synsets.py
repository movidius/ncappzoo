import csv
import os
import tarfile

count = 0

# CSV file with the synsets codes to untar
csv_file = csv.reader(open('./found_synsets.csv', "r"), delimiter=",")

# Base for ImageNet child tar files
child_tar_base = "./fall11_whole"

dir = os.mkdir('train_all')

for synset in csv_file:
    # If child synset found, decompress it
    if int(synset[1]) == 1:
        child_tar = os.path.join(child_tar_base, synset[0])
        child_tar = child_tar + ".tar"
        tar = tarfile.open(child_tar)
        tar.extractall(path=dir)
        tar.close()
        count += 1
#Report how many tar files were extracted
print ("Extracted tar directories:", count)
