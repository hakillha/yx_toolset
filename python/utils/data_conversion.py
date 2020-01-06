import json
import math
import os
import random

from collections import defaultdict
from glob import glob
from os.path import join as pj

def add_item(in_list, out_dict):
    for item in in_list:
        try:
            out_dict[os.path.basename(item).split('.')[0]].append(item)
        except:
            out_dict[os.path.basename(item).split('.')[0]] = [item]

def parse_folder_ft_det(input_path, subfolders, valid_ratio):
    """
    Point of having this as a separate interface is that
    we can easily deal with changes of file structure.

    Args:

    Returns:
        res_dict: {data_key: [abs_img_path, abs_label_path], ...}
        shuffled_list: list of keys from above dict but shuffled to index it 
    """
    if subfolders or isinstance(input_path, list):
        if subfolders:
            folder_list = [item for item in os.listdir(input_path) if os.path.isdir(pj(input_path, item))]
        elif isinstance(input_path, list):
            folder_list = input_path
        img_list = []
        anno_list = []
        for folder in folder_list:
            if isinstance(input_path, list):
                img_list += glob(pj(folder, 'images', '*'))
                anno_list += glob(pj(folder, 'labels', '*'))
            else:
                img_list += glob(pj(input_path, folder, 'images', '*')) # TODO: file type check
                anno_list += glob(pj(input_path, folder, 'labels', '*'))
            # img_list += glob(pj(input_path, folder, 'images', '*')) # TODO: file type check
            # anno_list += glob(pj(input_path, folder, 'labels', '*'))
    else:
        img_list = glob(pj(input_path, 'images', '*'))
        anno_list = glob(pj(input_path, 'labels', '*'))
    img_list = [im for im in img_list if im.endswith('.jpg')]
    anno_list = [ann for ann in anno_list if ann.endswith('.json')]
    # returns a dict where each entry is a list: [abspath_to_img, abspath_to_anno]
    res_dict = {}
    add_item(img_list, res_dict)
    add_item(anno_list, res_dict)
    for k, v in res_dict.items():
        # remove that from the dict if one file(img/ann) is missing
        if len(v) == 1:
            del res_dict[k]

    shuffled_list = list(res_dict.keys())
    random.shuffle(shuffled_list)
    train_size = int(math.floor(len(shuffled_list) * (1 - float(valid_ratio))))

    return res_dict, shuffled_list, train_size

def parse_cats_from_annos(data_map):
    cats = []
    cats_cnt = defaultdict(int)
    if isinstance(data_map, dict):
        data_map = [v[1] for k, v in data_map.items()]
    # for k, v in data_map.items():
        # with open(v[1]) as ann_file:
    for f in data_map:
        with open(f) as ann_file:
            json_dict = json.load(ann_file)
            if 'object' in json_dict['outputs'].keys():
                for obj in json_dict['outputs']['object']:
                    try:
                        cls_name = str(obj['name'])
                    except:
                        continue
                    cats_cnt[cls_name] += 1
    cats = cats_cnt.keys()
    cats.sort()
    cats_print = [(cat, cats_cnt[cat]) for cat in cats]
    print(cats)
    print(cats_print)
    ret = dict()
    for idx, cat in enumerate(cats):
        ret[cat] = idx + 1
    return ret

def find_det_parent_class(in_cls, finegrained_cls):
    if finegrained_cls:
        if in_cls.startswith('pl'):
            return 'pl'
        elif in_cls.startswith('pm'):
            return 'pm'
        elif in_cls.startswith('ph'):
            return 'ph'
        else:
            return in_cls
    else:
        if in_cls.startswith('p'):
            return 'p'
        if in_cls.startswith('i'):
            return 'i'
        return in_cls

def ft_mask_conversion(width, height, anno):
    xmin = width
    ymin = height
    xmax = 0
    ymax = 0
    point_set = []
    # if anno is a dictionary
    for i in range(int(len(anno['polygon']) / 2)):
        x = anno['polygon']['x' + str(i + 1)]
        y = anno['polygon']['y' + str(i + 1)]
        if(x < 0):
            x = 0
        if(x > width):
            x = width
        if(y < 0):
            y = 0
        if (y > height):
            y = height
        xmin = min(x, xmin)
        ymin = min(y, ymin)
        xmax = max(x, xmax)
        ymax = max(y, ymax)
        point_set.append(x)
        point_set.append(y)

    # A simple check to detect the illegal segementation annotation
    if len(point_set) <= 4:
        print('Illegal segmentation annotation!')
        return

    return [xmin, ymin, xmax, ymax], point_set

# PREDEFINED_CLASSES_GENERIC = ['i','p', 'wo', 'rn', 'lo', 'tl',  'ro']
PRE_DEFINE_CATEGORIES_GENERIC = {'i': 1, 'p': 2, 'wo': 3, 'rn': 4, 'lo': 5, 
                                 'tl': 6, 'ro': 7}
PREDEFINED_CLASSES = ['io', 'wo', 'ors', 'p10', 'p11', 
                      'p26', 'p20', 'p23', 'p19', 'pne',
                      'rn', 'ps', 'p5', 'lo', 'tl',
                      'pg', 'sc1','sc0', 'ro', 'pn',
                      'po', 'pl', 'pm']

def merge_cls(inFile, outFile):
    with open(inFile) as f:
        json_list = json.load(f)
    out_json_list = []
    for ann in json_list:
        if ann['category_name'] in ['ors', 'sc0', 'sc1']:
            continue
        elif ann['category_name'].startswith('p'):
            ann['category_name'] = 'p'
            ann['category_id'] = PRE_DEFINE_CATEGORIES_GENERIC['p']
        elif ann['category_name'] == 'io':
            ann['category_name'] = 'i'
            ann['category_id'] = PRE_DEFINE_CATEGORIES_GENERIC['i']
        else:
            ann['category_id'] = PRE_DEFINE_CATEGORIES_GENERIC[ann['category_name']]
        out_json_list.append(ann)
    with open(outFile, 'w') as f:
        f.write(json.dumps(out_json_list))

if __name__ == '__main__':
    merge_cls('/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/ignore_toosmall/11_30/og_files/generic_valid_sizethr625_fpn.json',
              '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/ignore_toosmall/11_30/og_files/merged_generic_valid_sizethr625_fpn.json')