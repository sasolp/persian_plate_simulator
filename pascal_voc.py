import glob
import os
import pickle
from tqdm import tqdm
from cv2 import cv2
import numpy
from json2xml import json2xml
import xml.etree.ElementTree as ET

kernel = numpy.ones((2, 2), numpy.uint8)

class ObjectPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class BoundBox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.width = w
        self.height = h


class PlateObject:
    def __int__(self):
        self.bounding_box: BoundBox = BoundBox(0, 0, 0, 0)
        self.lable = ''
        self.corners = []

    def __init__(self, lable, x, y, w, h, points: numpy.ndarray):
        self.bounding_box: BoundBox = BoundBox(x, y, w, h)
        self.lable = lable
        self.corners = points





def get_rects(img, threshold, single_instance) -> [tuple]:
    bbox_list = []
    thresh = (img == threshold).astype(numpy.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)  # noise removal steps
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours[0]:
        x, y, w, h = cv2.boundingRect(cnt)
        bbox_list.append((max(x - 2, 0), max(y - 2, 0), x + w - 2, y + h - 2))  # compensate dilation effect

    if single_instance and len(bbox_list) != 0:
        xmin = min([elem[0] for elem in bbox_list])
        ymin = min([elem[1] for elem in bbox_list])
        xmax = max([elem[2] for elem in bbox_list])
        ymax = max([elem[3] for elem in bbox_list])
        bbox_list.clear()
        bbox_list.append((xmin, ymin, xmax, ymax))
    return bbox_list


def bounding_boxs_to_xml(file_path, src_width, src_height, objects: numpy.ndarray, have_to_write_points = False):
    print(f'\ngenerating xml {file_path}')
    if objects is None:
        return
    real_path = os.path.realpath(file_path)
    path = real_path.split('\\')
    data_dic = {
        'folder': path[-2],
        'path': real_path,
        'filename': path[-1],
        'source': {
            'database': 'automated_lp'  # awfully hardcoded!
        },
        'size': {
            'width': src_width,
            'height': src_height,
            'depth': 3  # awfully hardcoded!
        },
        'segmented': 0  # awfully hardcoded!
    }
    index = 0
    data_dic['object'] = []
    for _object in objects:
        data_dic['object'].append({})
        data_dic['object'][index]['name'] = _object.lable
        data_dic['object'][index]['pose'] = 'unspecified'  # awfully hardcoded!
        data_dic['object'][index]['truncated'] = 0  # awfully hardcoded!
        data_dic['object'][index]['difficult'] = 0  # awfully hardcoded!
        data_dic['object'][index]['bndbox'] = {
            'xmin': _object.bounding_box.x,
            'ymin': _object.bounding_box.y,
            'xmax': _object.bounding_box.x + _object.bounding_box.width,
            'ymax': _object.bounding_box.y + _object.bounding_box.height
        }
        if have_to_write_points:
            points = []
            for pt in _object.corners:
                points.append([pt.x, pt.y])
            data_dic['object'][index]['pnts'] = {'pt': points}
        index += 1
    xml = json2xml.Json2xml(data_dic, wrapper='annotation').to_xml()
    with open(file_path, "w") as f_out:
        f_out.write(xml)


def parse_voc_annotation(ann_dir, img_dir, cache_name, labels=[]):
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        all_insts = []
        seen_labels = {}

        numbr = 0;

        # for ann in sorted(os.listdir(ann_dir)):
        for ann in sorted(os.listdir(img_dir)):
            if (ann.find('.jpg') < 0):
                continue;
            ann = ann.replace('.jpg', '.xml')
            print('=>>>>>>' + ann)
            img = {'object': []}

            try:
                tree = ET.parse(ann_dir + ann)
            except Exception as e:
                print(e)
                print('Ignore this bad annotation: ' + ann_dir + ann)
                continue

            for elem in tree.iter():
                if 'filename' in elem.tag:
                    elem.text = ann.replace('.xml', '.jpg')
                    img['filename'] = img_dir + elem.text
                if 'width' in elem.tag:
                    img['width'] = int(elem.text)
                if 'height' in elem.tag:
                    img['height'] = int(elem.text)
                if 'object' in elem.tag or 'part' in elem.tag:
                    obj = {}

                    for attr in list(elem):
                        if 'name' in attr.tag:
                            obj['name'] = attr.text

                            if obj['name'] in seen_labels:
                                seen_labels[obj['name']] += 1
                            else:
                                seen_labels[obj['name']] = 1

                            if len(labels) > 0 and obj['name'] not in labels:
                                break
                            else:
                                img['object'] += [obj]

                        if 'bndbox' in attr.tag:
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    obj['xmin'] = int(round(float(dim.text)))
                                if 'ymin' in dim.tag:
                                    obj['ymin'] = int(round(float(dim.text)))
                                if 'xmax' in dim.tag:
                                    obj['xmax'] = int(round(float(dim.text)))
                                if 'ymax' in dim.tag:
                                    obj['ymax'] = int(round(float(dim.text)))

            if len(img['object']) > 0:
                all_insts += [img]
                print(numbr)
                numbr += 1;

        cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_insts, seen_labels

def parse_voc_annotation2(img_dir):

    all_insts = []
    seen_labels = {}

    numbr = 0

    # for ann in sorted(os.listdir(ann_dir)):
    for img_name in sorted(os.listdir(img_dir)):
        if img_name.find('.png') < 0:
            continue
        img = {'object': []}
        img['filename'] = os.path.join(img_dir, img_name)
        image = cv2.imread(img['filename'] )
        image_h, image_w, _ = image.shape
        img['width'] = image_w
        img['height'] = image_h
        obj = {}
        obj['name'] = 'Plate'
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        bg = (img_gray > 0).astype(numpy.uint8)
        bg[0: int(0.75* image_h), :] = 0
        cv2.erode(bg, numpy.ones((5,5), dtype=numpy.uint8), iterations=1)
        contours = cv2.findContours(bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours[0]:
            if cv2.contourArea(cnt) < 300:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            bg_w = w
            offset_x = x
            break
        # cv2.imshow('0', bg * 255)
        # cv2.imshow('1', image)
        # cv2.waitKey()
        plate_width = int(numpy.random.randint(bg_w * 0.3, bg_w * 0.5))
        plate_height = int(numpy.random.randint(plate_width * 0.2, plate_width * 0.27))

        obj['xmin'] = offset_x + int((bg_w - plate_width) * 0.5)
        obj['ymin'] = int((image_h - plate_height*2))
        obj['xmax'] = offset_x + int(obj['xmin'] + plate_width)
        obj['ymax'] = int(obj['ymin'] + plate_height)
        img['object'] += [obj]

        all_insts += [img]
    return all_insts

def bounding_rects_to_xml(input_directory, output_directory, annotations_config):
    print('\ngenerating xmls...')
    file_names = glob.glob(input_directory)
    for file in tqdm(file_names):
        real_path = os.path.realpath(file)
        path = real_path.split('/')
        annotation = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        data_dic = {
            'folder': path[-2],
            'path': real_path,
            'filename': path[-1],
            'source': {
                'database': 'automated_lp'  # awfully hardcoded!
            },
            'size': {
                'width': annotation.shape[0],
                'height': annotation.shape[1],
                'depth': 3  # awfully hardcoded!
            },
            'segmented': 0  # awfully hardcoded!
        }
        index = 0
        data_dic['object'] = []
        for key in annotations_config:
            rects_list = get_rects(annotation, annotations_config[key][0], single_instance=annotations_config[key][1])
            for _object in rects_list:
                data_dic['object'].append({})
                data_dic['object'][index]['name'] = key
                data_dic['object'][index]['pose'] = 'unspecified'  # awfully hardcoded!
                data_dic['object'][index]['truncated'] = 0  # awfully hardcoded!
                data_dic['object'][index]['difficult'] = 0  # awfully hardcoded!
                data_dic['object'][index]['bndbox'] = {
                    'xmin': _object[0],
                    'ymin': _object[1],
                    'xmax': _object[2],
                    'ymax': _object[3]
                }
                index += 1
        xml = json2xml.Json2xml(data_dic, wrapper='annotation').to_xml()
        with open(os.path.join(output_directory, path[-1].replace('.png', '.xml')), "w") as f_out:
            f_out.write(xml)
