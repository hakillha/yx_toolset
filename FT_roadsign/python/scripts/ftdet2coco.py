from __future__ import absolute_import

import argparse
import cv2
import json
import math
import os
import random
import sys

from os.path import join as pj

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, pj(cur_path, '..', '..', '..', '..'))
import yx_toolset.python.utils.data_conversion as data_conversion

# PRE_DEFINE_CATEGORIES = None
# PRE_DEFINE_CATEGORIES = {'i': 1, 'p': 2, 'wo': 3, 'rn': 4, 'lo': 5, 
#                          'tl': 6, 'ro': 7}
PRE_DEFINE_CATEGORIES = {'io': 1, 'wo': 2, 'ors': 3, 'p10': 4, 'p11': 5, 
                         'p26': 6, 'p20': 7, 'p23': 8, 'p19': 9, 'pne': 10, 
                         'rn': 11, 'ps': 12, 'p5': 13, 'lo': 14, 'tl': 15, 
                         'pg': 16, 'sc1': 17,'sc0': 18, 'ro': 19, 'pn': 20, 
                         'po': 21, 'pl': 22, 'pm': 23}

def parse_args():
    parser = argparse.ArgumentParser('Convert FT detection data into COCO format.')
    parser.add_argument('--input_path', default='/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup', type=str)
    parser.add_argument('--subfolders', help='Indicates if the input path contains subfolders.',action='store_true')
    parser.add_argument('--train_json_file', default='/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/cocoformat_train_out.json', type=str)
    parser.add_argument('--valid_json_file', default='/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/cocoformat_valid_out.json', type=str)
    parser.add_argument('--valid_ratio', default=0.15, help='The ratio of validation files.', type=float)
    return parser.parse_args()

def get_categories(in_files):
    classes_names = set()

    for index in in_files.keys():
        json_dict = json.load(open(in_files[index][1]))
        if 'outputs' in json_dict.keys():
            if 'object' in json_dict['outputs'].keys():
                for bb in json_dict['outputs']['object']:
                    classes_names.add(bb['name'])

    return {name: i + 1 for i, name in enumerate(classes_names)}

def find_parent_class(in_cls, orphan=True):
    if orphan:
        if in_cls.startswith('pl'):
            return 'pl'
        elif in_cls.startswith('pm'):
            return 'pm'
        else:
            return in_cls
    else:
        if in_cls.startswith('p'):
            return 'p'
        if in_cls.startswith('i'):
            return 'i'
        return in_cls

def detect_bad_data(json_dict, classes_names):
    '''
        Returns a warning string if an error is found, otherwise returns None.
    '''
    if json_dict['labeled'] == False:
        return 'Image {} is not labeled.'.format(index)
    if 'outputs' in json_dict.keys():
        if 'object' in json_dict['outputs'].keys():
            contain_coi = False
            for obj in json_dict['outputs']['object']:
                if find_parent_class(obj['name']) in classes_names.keys():
                    contain_coi = True                     
            if not contain_coi:
                return 'This frame doesn\'t contain classes of interest.'
        else:
            return 'No object in the file.'
    else:
        return 'No output in the file.'

    return None

def write_one_image(json_dict, 
                    out_json_dict, 
                    categories, 
                    anno_id, 
                    image_path, 
                    img_id,
                    output_irre=False):
    """
    Args:
        output_irre: If set to true save irrelavant files instead
    """
    bad_flag = False
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    recorded_width = json_dict['size']['width']
    recorded_height = json_dict['size']['height']
    if height != recorded_height:
        # print('Warning: Inconsistent image heights! Image file: {}. Anno file: {}.'.format(height, recorded_height))
        bad_flag = True
    if width != recorded_width:
        # print('Warning: Inconsistent image widths! Image file: {}. Anno file: {}.'.format(width, recorded_width))
        bad_flag = True
    img_id += 1
    img_info = {
        'file_name': os.path.basename(image_path),
        'height': height,
        'width': width,
        'id': img_id
    }

    # paste this whenever you want to visualize the json record
    # print(json.dumps(json_dict, indent=4))
    valid_anno = False
    for anno in json_dict['outputs']['object']:
        if find_parent_class(anno['name']) in categories.keys() and 'bndbox' in anno.keys():
            cate_id = categories[find_parent_class(anno['name'])]
            anno = anno['bndbox']
            bb = [anno['xmin'], anno['ymin'], anno['xmax'], anno['ymax']]
            if not (bb[0] >= 0 and bb[1] >= 0 and bb[2] < width and bb[3] < height):
                # print('Annotation out of boundary!')
                # print('[xmin, ymin, width, height]: ' + str(bb))
                # print('Image width, height: ' + str((width, height)))

                # This fix is under the assumption
                # the bbox is not way too off the frame
                bb[0] = max(0, bb[0])
                bb[1] = max(0, bb[1])
                bb[2] = min(width, bb[2])
                bb[3] = min(height, bb[3])
                bad_flag = True
            bb = [bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]]
            # assert bb[0] >= 0 and bb[1] >= 0 and bb[0] + bb[2] < width and bb[1] + bb[3] < height, 'Annotation out of boundary!'
            area = bb[2] * bb[3]
            anno_entry = {
                'area': area,
                'bbox': bb,
                'iscrowd': 0,
                'image_id': img_id,
                'category_id': cate_id,
                'id': anno_id,
                'ignore': 0,
            }

            out_json_dict['annotations'].append(anno_entry)
            valid_anno = True
            anno_id += 1

    if valid_anno == 0:
        return

    # Not appending this record if there is no valid anno in it
    out_json_dict['images'].append(img_info)

    return  img_id, anno_id, bad_flag

def convert(data_map, data_map_keys, out_file, categories):
    out_json_dict = {'images': [], 'type': 'instances', 'annotations':[], 'categories':[]}

    img_id = 1
    anno_id = 1
    irre_data_cnt = 0
    invalid_data_cnt = 0
    bad_image_data = 0
    for key in data_map_keys:
        print('{}/{} finished...'.format(img_id, len(data_map_keys)))
        json_dict = json.load(open(data_map[key][1]))
        bad_res = detect_bad_data(json_dict, categories)
        if bad_res:
            # print(bad_res)
            invalid_data_cnt += 1
            continue

        ret = write_one_image(json_dict, 
                              out_json_dict, 
                              categories, 
                              anno_id, 
                              data_map[key][0], 
                              img_id)
        if ret:
            img_id, anno_id, _ = ret
            if ret[2]: bad_image_data += 1
        else:
            irre_data_cnt += 1

    for cate, cid in categories.items():
        cate_entry = {'supercategory': 'none', 'id': cid, 'name': cate}
        out_json_dict['categories'].append(cate_entry)
    # sort it in the out dict
    out_json_dict['categories'].sort(key=lambda val: val['id'])

    out_f = open(out_file, 'w')
    out_f.write(json.dumps(out_json_dict))
    out_f.close()

    return irre_data_cnt, bad_image_data, invalid_data_cnt

if __name__ == '__main__':
    args = parse_args()

    data_map, shuffled_list, train_size = data_conversion.parse_folder_ft_det(args.input_path, args.subfolders, args.valid_ratio)
    if PRE_DEFINE_CATEGORIES == None:
        categories = get_categories(data_map)
    else:
        categories = PRE_DEFINE_CATEGORIES

    irre_data_cnt = 0
    bad_image_data = 0 
    invalid_data_cnt = 0
    if args.valid_ratio == 1:
        stat = convert(data_map, shuffled_list, args.valid_json_file, categories)
        irre_data_cnt += stat[0]
        bad_image_data += stat[1]
        invalid_data_cnt += stat[2]
    elif args.valid_ratio == 0:
        stat = convert(data_map, shuffled_list, args.train_json_file, categories)
        irre_data_cnt += stat[0]
        bad_image_data += stat[1]
        invalid_data_cnt += stat[2]
    else:
        stat = convert(data_map, shuffled_list[:train_size], args.train_json_file, categories)
        irre_data_cnt += stat[0]
        bad_image_data += stat[1]
        invalid_data_cnt += stat[2]
        stat = convert(data_map, shuffled_list[train_size:], args.valid_json_file, categories)
        irre_data_cnt += stat[0]
        bad_image_data += stat[1]
        invalid_data_cnt += stat[2]
    print('# of files without bbox data entry: ' + str(irre_data_cnt))
    print('# of files with corrupted data: ' + str(bad_image_data))
    print('# of invalid files: ' + str(invalid_data_cnt))