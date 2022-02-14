import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET



MY_PATH = '/Users/yassinedehbi/'
#my_mapping = {'SITTOR':0, 'PINARB':1, 'ROUGOR':2, 'MESCHA':3, 'MESNON':4, 'ACCMOU':5, 'MESBLE':6}
def parse_doc(xml_file:str):
    final_list = []
    root =  ET.parse(xml_file).getroot()
    l = root.findall("image")
    for element in l:        
        image_dict = {}
        pathh = element.attrib['name']
        k = pathh.split('/')
        k[0] = MY_PATH+'ProjetLong/OrnithoMateData/raw_data'
        separator = '/'
        pathh = separator.join(k)
        image_dict['path'] = pathh
        image_dict['width'] = element.attrib['width']
        image_dict['height'] = element.attrib['height']
        image_dict['elements'] = []
        boxes = element.findall("box")
        box_dict = {}
        for box in boxes:
            for attribue in box.findall('.//attribute'):
                if attribue.attrib['name']=="species":
                    if attribue.text == 'noBird' or attribue.text == 'unknown' or attribue.text == 'human':
                        pass
                    else :

                        #box_dict["label"] = my_mapping[attribue.text]
                        box_dict["label"] = attribue.text
                        box_dict["xtl"] = box.attrib["xtl"]
                        box_dict["ytl"] = box.attrib["ytl"]
                        box_dict["xbr"] = box.attrib["xbr"]
                        box_dict["ybr"] = box.attrib["ybr"]
            
                        image_dict['elements'].append(box_dict)
                else:
                    pass
        final_list.append(image_dict)
    return final_list

l = []
for xml_file in os.listdir(MY_PATH+'ProjetLong/OrnithoMateData/cvat_annotations_wip'):
    ll = parse_doc(MY_PATH+'ProjetLong/OrnithoMateData/cvat_annotations_wip/'+xml_file)
    l+=ll
for xml_file in os.listdir(MY_PATH+'ProjetLong/OrnithoMateData/cvat_annotations_copy'):
    ll = parse_doc(MY_PATH+'ProjetLong/OrnithoMateData/cvat_annotations_copy/'+xml_file)
    l+=ll

labels = ['SITTOR','PINARB','TOUTUR','ROUGOR','MESCHA','MOIDOM','MESNON','VEREUR','ACCMOU','MESBLE']

my_mapping = {labels[i]:i for i in range(len(labels))}

data_file = open(MY_PATH+'ProjetLong/data.txt', 'w')
for img in l:
    line = img['path'] + ' '
    if len(img['elements']) !=0:
        for element in img['elements']:
            line += element['xtl'] + ',' + element['ytl'] + ',' + element['xbr'] + ',' +  element['ybr'] + ','+ str(my_mapping[element['label']]) + ' '
        data_file.write(line+'\n')

with open(MY_PATH+'ProjetLong/data.txt', 'r') as dl:
    lines = dl.readlines()
#lines
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
total = len(lines)
train_lines = lines[:int(len(lines)*0.8)]
val_lines = lines[int(len(lines)*0.8):int(len(lines)*0.9)]
test_lines = lines[int(len(lines)*0.9):]
train_file = open(MY_PATH+'ProjetLong/data/train_data.txt', 'w')
val_file = open(MY_PATH+'ProjetLong/data/val_data.txt', 'w')
test_file = open(MY_PATH+'ProjetLong/data/test_data.txt', 'w')
train_file.writelines(train_lines)
val_file.writelines(val_lines)
test_file.writelines(test_lines)
train_file.close()
test_file.close()
val_file.close()
