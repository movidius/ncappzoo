# Filter out annotations without the objects belong to the defined 9 classes.
# The filtered annotations will be put under Annotations_x10 in data/VOCdevkit/VOC2007 and ./data/VOCdevkit/VOC2012

import xml.etree.ElementTree as ET
import os

X10_LABELS = ['bicycle', 'bird', 'bus', 'car', 'cat',
              'dog', 'motobike', 'person', 'pottedplant']

def check_obj_class_in_table(obj, labels):
    name = obj.find('name')
    if name.text in labels:
        return True
    else:
        return False

def filter_xmlfiles(input_xml, output_xml):

    tree = ET.parse(input_xml)
    root = tree.getroot()

    obj_exist = False
    for obj in root.findall('object'):
        if not check_obj_class_in_table(obj, X10_LABELS):
            root.remove(obj)
        else:
            obj_exist = True

    # if any object exists in xml
    if obj_exist:
        tree.write(output_xml)

def filter_voc_folder(input_path, out_path):

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for xfile in os.listdir(input_path):
        infile = os.path.join(input_path, xfile)
        outfile = os.path.join(out_path, xfile)
        filter_xmlfiles(infile, outfile)

if __name__ == "__main__":

    voc_path =  "./data/VOCdevkit"
    filter_voc_folder(os.path.join(voc_path, 'VOC2007', 'Annotations'),
                      os.path.join(voc_path, 'VOC2007', 'Annotations_x10'))

    filter_voc_folder(os.path.join(voc_path, 'VOC2012', 'Annotations'),
                      os.path.join(voc_path, 'VOC2012', 'Annotations_x10'))
