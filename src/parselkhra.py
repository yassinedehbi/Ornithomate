
import xml.etree.ElementTree as ET
def parse_doc(xml_file:str):
    final_list = []
    root =  ET.parse(xml_file).getroot()
    l = root.findall("image")
    for element in l:        
        image_dict = {}
        image_dict['path'] = element.attrib['name']
        image_dict['width'] = element.attrib['width']
        image_dict['height'] = element.attrib['height']
        image_dict['elements'] = []
        boxes = element.findall("box")
        box_dict = {}
        for box in boxes:
            box_dict["xtl"] = box.attrib["xtl"]
            box_dict["ytl"] = box.attrib["ytl"]
            box_dict["xbr"] = box.attrib["xbr"]
            box_dict["ybr"] = box.attrib["ybr"]
            for attribue in box.findall('.//attribute'):
                if attribue.attrib['name']=="species":
                    box_dict["label"] = attribue.text
        image_dict['elements'].append(box_dict)
        final_list.append(image_dict)
    return final_list

#exemple
#[{'path': 'bird/task_05-01-2021/2021-01-05-18-17-53.jpg', 'width': '1920', 'height': '1088', 'elements': [{'xtl': '15.48', 'ytl': '15.28', 'xbr': '1908.46', 'ybr': '1080.40', 'label': 'noBird'}]}, {'path': 'bird/task_05-01-2021/2021-01-05-18-17-51.jpg', 'width': '1920', 'height': '1088', 'elements': [{'xtl': '11.62', 'ytl': '9.49', 'xbr': '1914.25', 'ybr': '1088.00', 'label': 'noBird'}]}, {'path': 'bird/task_05-01-2021/2021-01-05-18-17-49.jpg', 'width': '1920', 'height': '1088', 'elements': [{'xtl': '7.76', 'ytl': '5.60', 'xbr': '1906.53', 'ybr': '1088.00', 'label': 'noBird'}]}, {'path': 'bird/task_05-01-2021/2021-01-05-18-17-47.jpg', 'width': '1920', 'height': '1088', 'elements': [{'xtl': '17.41', 'ytl': '15.28', 'xbr': '1904.60', 'ybr': '1080.40', 'label': 'noBird'}]}]


        




