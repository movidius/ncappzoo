import argparse
import os

def get_str_from_path(xfile):
    xlist = xfile.split('/')[-3:]
    return os.path.join(xlist[0], xlist[1], xlist[2])

def generate_file_list(inputfile, txtfile, imagepath, xmlpath, append):

    with open(inputfile) as f:
        idlist = f.readlines()

    xmlfilelist = os.listdir(xmlpath)

    if append:
        f =  open(txtfile, 'a')
    else:
        f =  open(txtfile, 'w')

    for strid in idlist:
        id = strid.strip()
        xml = id+'.xml'
        if xml in xmlfilelist:
            xmlfile = os.path.abspath(os.path.join(xmlpath, id+'.xml'))
            imgfile = os.path.abspath(os.path.join(imagepath, id+'.jpg'))
            f.write(get_str_from_path(imgfile)+' '+get_str_from_path(xmlfile)+'\n')

    f.close()

    print 'File list saved to ' + txtfile

if __name__ == "__main__":


    file_path    =  "data"
    voc2012_path =  "data/VOCdevkit/VOC2012"
    voc2007_path =  "data/VOCdevkit/VOC2007"

    # VOC2012, trainval
    generate_file_list(os.path.join(voc2012_path, 'ImageSets', 'Main', 'trainval.txt'),
                       os.path.join(file_path, 'trainval.txt'),
                       os.path.join(voc2012_path, 'JPEGImages'),
                       os.path.join(voc2012_path, 'Annotations_x10'), append=False)

    # VOC2007, trainval
    generate_file_list(os.path.join(voc2007_path, 'ImageSets', 'Main', 'trainval.txt'),
                       os.path.join(file_path, 'trainval.txt'),
                       os.path.join(voc2007_path, 'JPEGImages'),
                       os.path.join(voc2007_path, 'Annotations_x10'), append=True)
    # VOC2007, test
    generate_file_list(os.path.join(voc2007_path, 'ImageSets', 'Main', 'test.txt'),
                       os.path.join(file_path, 'test.txt'),
                       os.path.join(voc2007_path, 'JPEGImages'),
                       os.path.join(voc2007_path, 'Annotations_x10'), append=False)