import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET

from os import path

MY_PATH = '/Users/akbya/Downloads/'
def parse_doc(xml_file:str):
    final_list = []
    root =  ET.parse(xml_file).getroot()
    image_dict = {}
    pathh = root.find("filename").text
    k = pathh.split('/')
    k[0] = MY_PATH+'Ornithomate/raw_data'
    separator = '/'
    pathh = separator.join(k)
    image_dict['path'] = pathh
    image_dict['width'] = root.find("size").find("width").text
    image_dict['height'] = root.find("size").find("height").text
    image_dict['box'] = {}
    for bird in root.findall("object"):
        image_dict['box']={}
        for attribue in bird.find('attributes'):
            if attribue.find('name').text=="species":
                if attribue.find('value').text != 'noBird' and attribue.find('value').text != 'unknown' and attribue.find('value').text != 'human':
                    image_dict['box']["label"] = attribue.find('value').text
                    image_dict['box']["xmin"] = bird.find("bndbox").find("xmin").text
                    image_dict['box']["ymin"] = bird.find("bndbox").find("ymin").text
                    image_dict['box']["xmax"] = bird.find("bndbox").find("xmax").text
                    image_dict['box']["ymax"] = bird.find("bndbox").find("ymax").text
        final_list.append(image_dict)
    return final_list

l = []
for file in os.listdir(MY_PATH+'OrnithoMate/annotations/'):
    for xml_file in os.listdir(MY_PATH+'OrnithoMate/annotations/'+file+'/Annotations/bird/'+file):
        ll = parse_doc(MY_PATH+'OrnithoMate/annotations/'+file+'/Annotations/bird/'+file+'/'+xml_file)
        if len(ll)>1:
            for dicti in ll:
                l+=[dicti]
        elif len(ll)==1:
            l+=ll

labels = ['SITTOR','PINARB','TOUTUR','ROUGOR','MESCHA','MOIDOM','MESNON','VEREUR','ACCMOU','MESBLE', 'ECUROU', 'PIEBAV', 'MULGRI',
        'CAMPAG', 'MESNOI', 'MESHUP', 'BERGRI']

my_mapping = {labels[i]:i for i in range(len(labels))}

train_val_file = open(MY_PATH+'Ornithomate/data.txt', 'w')
test_file = open(MY_PATH+'Ornithomate/test_data.txt', 'w')
tasks = pd.read_csv('/Users/akbya/Downloads/Ornithomate/Ornithotasks-CVAT_task.csv', header=None)
tasks.columns = ["Task", "Site", "Date", "Day2", "Split"]
train_tasks = []
test_tasks = []
for index, row in tasks.iterrows():
    if row["Split"]=="TRAIN":
        train_tasks.append(row["Task"])
    elif row["Split"]=="TEST":
        test_tasks.append(row["Task"])

for img in l:
    line = img['path'] + ' '
    if os.path.basename(os.path.dirname(img['path'])) in train_tasks:
        if len(img['box']) !=0:
            line += img['box']['xmin'] + ',' + img['box']['ymin'] + ',' + img['box']['xmax'] + ',' +  img['box']['ymax'] + ','+ str(my_mapping[img['box']['label']])
            train_val_file.write(line+'\n')
    elif os.path.basename(os.path.dirname(img['path'])) in test_tasks:
        if len(img['box']) !=0:
            line += img['box']['xmin'] + ',' + img['box']['ymin'] + ',' + img['box']['xmax'] + ',' +  img['box']['ymax'] + ','+ str(my_mapping[img['box']['label']])
            test_file.write(line+'\n')


with open(MY_PATH+'Ornithomate/data.txt', 'r') as dl:
    lines = dl.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
total = len(lines)
train_lines = lines[:int(len(lines)*0.8)]
val_lines = lines[int(len(lines)*0.8):]
train_file = open(MY_PATH+'Ornithomate/train_data.txt', 'w')
train_file.writelines(train_lines)
val_file = open(MY_PATH+'Ornithomate/val_data.txt', 'w')
val_file.writelines(val_lines)
train_val_file.close()
test_file.close()
val_file.close()