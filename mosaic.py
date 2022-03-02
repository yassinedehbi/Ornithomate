import random

import cv2
import os
import glob
import numpy as np
from PIL import Image

import random

import cv2
import os
import glob
import numpy as np
from PIL import Image

OUTPUT_SIZE = (600, 600)  # Height, Width
SCALE_RANGE = (0.3, 0.7)
FILTER_TINY_SCALE = 1 / 50  # if height or width lower than this scale, drop it.

ANNO_DIR = '/path/to/annotation/directory'
IMG_DIR = '/path/to/image/directory'

# Names of all the classes as they appear in the Pascal VOC Dataset
category_name = ['0','1','2','3','4','5','6','7','8','9']

def mosaic(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale=0.):
    output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
    scale_x = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    scale_y = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    divid_point_x = int(scale_x * output_size[1])
    divid_point_y = int(scale_y * output_size[0])

    new_anno = []
    for i, idx in enumerate(idxs):
        path = all_img_list[idx]
        img_annos = all_annos[idx]

        img = cv2.imread(path)
        if i == 0:  # top-left
            img = cv2.resize(img, (divid_point_x, divid_point_y))
            output_img[:divid_point_y, :divid_point_x, :] = img
            for bbox in img_annos:
                
                # As YOLO annotations have different centers from the image, this is how the bbox coordinates are calculated
                xmin = bbox[1] - (bbox[3]-bbox[1])*0.5
                ymin = bbox[2] - (bbox[4]-bbox[2])*0.5
                xmax = bbox[1] + (bbox[3]-bbox[1])*0.5
                ymax = bbox[2] + (bbox[4]-bbox[2])*0.5

                xmin *= scale_x
                ymin *= scale_y
                xmax *= scale_x
                ymax *= scale_y
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])

        elif i == 1:  # top-right
            img = cv2.resize(img, (output_size[1] - divid_point_x, divid_point_y))
            output_img[:divid_point_y, divid_point_x:output_size[1], :] = img
            for bbox in img_annos:
                xmin = bbox[1] - bbox[3]*0.5
                ymin = bbox[2] - bbox[4]*0.5
                xmax = bbox[1] + bbox[3]*0.5
                ymax = bbox[2] + bbox[4]*0.5

                xmin = scale_x + xmin * (1 - scale_x)
                ymin = ymin * scale_y
                xmax = scale_x + xmax * (1 - scale_x)
                ymax = ymax * scale_y
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
        elif i == 2:  # bottom-left
            img = cv2.resize(img, (divid_point_x, output_size[0] - divid_point_y))
            output_img[divid_point_y:output_size[0], :divid_point_x, :] = img
            for bbox in img_annos:
                xmin = bbox[1] - bbox[3]*0.5
                ymin = bbox[2] - bbox[4]*0.5
                xmax = bbox[1] + bbox[3]*0.5
                ymax = bbox[2] + bbox[4]*0.5

                xmin = xmin * scale_x
                ymin = scale_y + ymin * (1 - scale_y)
                xmax = xmax * scale_x
                ymax = scale_y + ymax * (1 - scale_y)
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
        else:  # bottom-right
            img = cv2.resize(img, (output_size[1] - divid_point_x, output_size[0] - divid_point_y))
            output_img[divid_point_y:output_size[0], divid_point_x:output_size[1], :] = img
            for bbox in img_annos:
                xmin = bbox[1] - bbox[3]*0.5
                ymin = bbox[2] - bbox[4]*0.5
                xmax = bbox[1] + bbox[3]*0.5
                ymax = bbox[2] + bbox[4]*0.5

                xmin = scale_x + xmin * (1 - scale_x)
                ymin = scale_y + ymin * (1 - scale_y)
                xmax = scale_x + xmax * (1 - scale_x)
                ymax = scale_y + ymax * (1 - scale_y)
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])

    if 0 < filter_scale:
        new_anno = [anno for anno in new_anno if
                    filter_scale < (anno[3] - anno[1]) and filter_scale < (anno[4] - anno[2])]

    return output_img, new_anno

# Change the output path of the imwrite commands to wherever you want to get both the mosaic image and the image with boxes
def main():
    #img_paths, annos = dataset(ANNO_DIR, IMG_DIR)
    img_paths = ['/Users/akbya/Downloads/Ornithomate/raw_data/task_21-01-2021/2021-01-21-14-18-13.jpg',
                '/Users/akbya/Downloads/Ornithomate/raw_data/task_05-01-2021/2021-01-05-17-05-13.jpg',
                '/Users/akbya/Downloads/Ornithomate/raw_data/task_06-01-2021/2021-01-06-12-13-46.jpg',
                '/Users/akbya/Downloads/Ornithomate/raw_data/task_21-01-2021/2021-01-21-12-34-23.jpg']
    annos = [[[4,1.97,200.53,577.00,970.46]],[[0,1308.30,435.95,1914.25,1036.07]],[[4, 0.00,549.80,937.85,1088.00]], [[9, 324.22,291.22,903.11,731.18]]]

    idxs = random.sample(range(len(annos)), 4)

    new_image, new_annos = mosaic(img_paths, annos,
    	                          idxs,
    	                          OUTPUT_SIZE, SCALE_RANGE,
    	                          filter_scale=FILTER_TINY_SCALE)

    cv2.imwrite('output.jpg', new_image) #The mosaic image
    for anno in new_annos:
        start_point = (int(anno[1] * OUTPUT_SIZE[1]), int(anno[2] * OUTPUT_SIZE[0]))
        end_point = (int(anno[3] * OUTPUT_SIZE[1]), int(anno[4] * OUTPUT_SIZE[0]))
        cv2.rectangle(new_image, start_point, end_point, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite('output_box.jpg', new_image) # The mosaic image with the bounding boxes
    
    yolo_anno = []
    
    for anno in new_annos:
      tmp = []
      tmp.append(anno[0])
      tmp.append((anno[3]+anno[1])/2)
      tmp.append((anno[4]+anno[2])/2)
      tmp.append(anno[3]-anno[1])
      tmp.append(anno[4]-anno[2])
      yolo_anno.append(tmp)

    with open('output.txt', 'w') as file: # The output annotation file will appear in the output.txt file
      for line in yolo_anno:
        file.write((' ').join([str(x) for x in line]) + '\n')   

if __name__ == '__main__':
    main()