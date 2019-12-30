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
from yx_toolset.python.utils.data_conversion import parse_cats_from_annos


PRE_DEFINE_CATEGORIES_GENERIC_UPDATED = {'i': 1, 'p': 2, 'wo': 3, 'rn': 4, 'lo': 5, 
                                         'tl': 6, 'ro': 7, 'sc0': 8, 'sc1': 9, 'ors': 10}
# PRE_DEFINE_CATEGORIES_GENERIC_UPDATED_TEST = {'i': 1, 'p': 2, 'wo': 3, 'rn': 4, 'lo': 5, 
#                                               'tl': 6, 'ro': 7, 'sc0': 8, 'sc1': 9}

# PRE_DEFINE_CATEGORIES_GENERIC = {'i': 1, 'p': 2, 'wo': 3, 'rn': 4, 'lo': 5, 
#                                  'tl': 6, 'ro': 7}
# PRE_DEFINE_CATEGORIES = {'io': 1, 'wo': 2, 'ors': 3, 'p10': 4, 'p11': 5, 
#                          'p26': 6, 'p20': 7, 'p23': 8, 'p19': 9, 'pne': 10, 
#                          'rn': 11, 'ps': 12, 'p5': 13, 'lo': 14, 'tl': 15, 
#                          'pg': 16, 'sc1': 17,'sc0': 18, 'ro': 19, 'pn': 20, 
#                          'po': 21, 'pl': 22, 'pm': 23}

# ['io', 'lo', 'tl', 'rn', 'ro', 
   # 'wo', 'ors', 'sc1', 'sc0', 'p26', 
   # 'p20', 'p23', 'ps', 'pne', 'pg', 
   # 'pn', 'po', 'pl', 'pm', 'p10', 
   # 'p11', 'p19', 'p5']

# PRE_DEFINE_CATEGORIES = {'io': 1, 'p26': 10, 'p20': 11, 'p23': 12, 'ps': 13, 
#                          'lo': 2, 'tl': 3, 'pg': 15, 'rn': 4, 'ro': 5, 
#                          'pn': 16, 'po': 17, 'pl': 18, 'pm': 19, 'wo': 6, 
#                          'ors': 7, 'sc1': 8, 'p10': 20, 'p11': 21, 'p19': 22, 
#                          'pne': 14, 'sc0': 9, 'p5': 23}

# PRE_DEFINE_CATEGORIES = {'sc0': 1, 'sc1': 2}

PRE_DEFINE_CATEGORIES = PRE_DEFINE_CATEGORIES_GENERIC_UPDATED

def get_categories(in_files):
    classes_names = set()

    for index in in_files.keys():
        with open(in_files[index][1]) as coco_ann:
            json_dict = json.load(coco_ann)
        if 'outputs' in json_dict.keys():
            if 'object' in json_dict['outputs'].keys():
                for bb in json_dict['outputs']['object']:
                    classes_names.add(bb['name'])
    return {name: i + 1 for i, name in enumerate(classes_names)}

def detect_bad_data(json_dict, categories):
    '''
        Returns a warning string if an error is found, otherwise returns None.
    '''
    if json_dict['labeled'] == False:
        # return 'Image {} is not labeled.'.format(json_dict['path'])
        return 'Image is not labeled.'
    if 'outputs' in json_dict.keys():
        if 'object' in json_dict['outputs'].keys():
            contain_coi = False
            for obj in json_dict['outputs']['object']:
                if obj['name'] in categories.keys():
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
                    set_type,
                    size_thresh,
                    output_irre=False):
    """
    Args:
        output_irre: If set to true save irrelavant files instead
    """
    bad_flag = False
    too_small = 0
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
        if anno['name'] in categories.keys() and 'bndbox' in anno.keys():
            cate_id = categories[anno['name']]
            bb = [int(anno['bndbox']['xmin']), 
                  int(anno['bndbox']['ymin']),
                  int(anno['bndbox']['xmax']),
                  int(anno['bndbox']['ymax'])]
            if not (bb[0] >= 0 and bb[1] >= 0 and bb[2] < width and bb[3] < height):
                # This fix is under the assumption
                # the bbox is not way too off the frame
                bb[0] = max(0, bb[0])
                bb[1] = max(0, bb[1])
                bb[2] = min(width, bb[2])
                bb[3] = min(height, bb[3])
                bad_flag = True
            bb = [bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]]
            area = bb[2] * bb[3]
            if set_type == 'valid' and area < size_thresh:
                too_small += 1
                continue
            anno_entry = {
                'area': area,
                'bbox': bb,
                'iscrowd': 0,
                'image_id': img_id,
                'category_id': cate_id,
                'category_name': anno['name'],
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

    return  img_id, anno_id, bad_flag, too_small

def convert(data_map, data_map_keys, out_file, categories, set_type, valid_size_thr=0):
    out_json_dict = {'images': [], 'type': 'instances', 'annotations':[], 'categories':[]}

    img_id = 1
    anno_id = 1
    irre_data_cnt = 0
    invalid_data_cnt = 0
    bad_image_data = 0
    num_too_small = 0
    for key in data_map_keys:
        print('{}/{} finished...'.format(img_id, len(data_map_keys)))
        with open(data_map[key][1]) as ann_file:
            json_dict = json.load(ann_file)
        # TODO: Put all data check here
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
                              img_id,
                              set_type,
                              valid_size_thr)
        if ret:
            img_id, anno_id, _, too_small = ret
            if ret[2]: bad_image_data += 1
            num_too_small += too_small
        else:
            irre_data_cnt += 1

    for cate, cid in categories.items():
        cate_entry = {'supercategory': 'none', 'id': cid, 'name': cate}
        out_json_dict['categories'].append(cate_entry)
    # sort it in the out dict
    out_json_dict['categories'].sort(key=lambda val: val['id'])

    with open(out_file, 'w') as out_f:
        out_f.write(json.dumps(out_json_dict))

    print('Num of anns that is too small: {}'.format(num_too_small))

    return irre_data_cnt, bad_image_data, invalid_data_cnt

def parse_args():
    parser = argparse.ArgumentParser("""Convert FT detection data to COCO format. 
                                        For ease of tracking we store the class_name/class_id list/dictionary in the code.
                                        So please look them up in the code before using this script 
                                        and change it according to your needs.""")
    parser.add_argument('--input_path', type=str,
                                        help="""Root of the data directory. 
                                        e.g. The folder that contains "images" and "labels" folders. """)
    parser.add_argument('--train_json_file', type=str,
                                             help="""Specify a path like "{PATH}/train.json". 
                                             This script itself doesn't create the folders so you need to create them yourself.""")
    parser.add_argument('--valid_json_file', type=str,
                                             help="""Specify a path like "{PATH}/valid.json". 
                                             This script itself doesn't create the folders so you need to create them yourself.""")
    parser.add_argument('--valid_size_thr', default=625, type=int, 
                                            help="""Any annotation that has an area smaller than this value will be ignored. 
                                            Set it to 0 to turn it off. This only applies to test set.""")
    parser.add_argument('--valid_ratio', default=0.15, type=float, 
                                         help="""The ratio of validation files. This also specify this script\'s behavior. e.g. 
                                         If this value is set to 1 or 0 
                                         then only the validation or train json file will be created respectively.""")
    parser.add_argument('--subfolders', action='store_true', 
                                        help="""Indicates if the input path contains subfolders 
                                        and each subfolder has its own "images", "labels" folders.
                                        Since we only care about the merged data files structure, 
                                        you can safely ignore this for now.""")
    parser.add_argument('--parse_cats_from_annos', default=True, type=bool, 
                                                   help="""If set to true the class dictionary will be parsed
                                                   from the annotation files.""")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    args.input_path = ['/media/yingges/Data_Junior/data/ft_pic/GIS-SIGN_optimize/annotation-20191121/device_didi',
                       '/media/yingges/Data_Junior/data/ft_pic/GIS-SIGN_arrows/arrows-20191107',
                       '/media/yingges/Data_Junior/data/ft_pic/GIS-SIGN_02/annotation-20191205',
                       '/media/yingges/Data_Junior/data/ft_pic/GIS-SIGN_02/online-20191128',
                       '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup'
                       ]

    data_map, shuffled_list, train_size = data_conversion.parse_folder_ft_det(args.input_path, args.subfolders, args.valid_ratio)
    if parse_cats_from_annos:
        categories = parse_cats_from_annos(data_map)
        categories_test = categories
    else:
        if PRE_DEFINE_CATEGORIES == None:
            categories = get_categories(data_map)
        else:
            # categories = PRE_DEFINE_CATEGORIES_GENERIC_UPDATED
            # categories_test = PRE_DEFINE_CATEGORIES_GENERIC_UPDATED_TEST
            categories = PRE_DEFINE_CATEGORIES
            categories_test = PRE_DEFINE_CATEGORIES

    irre_data_cnt = 0
    bad_image_data = 0 
    invalid_data_cnt = 0
    if args.valid_ratio == 1:
        stat = convert(data_map, shuffled_list, args.valid_json_file, categories_test, 'valid', args.valid_size_thr)
        irre_data_cnt += stat[0]
        bad_image_data += stat[1]
        invalid_data_cnt += stat[2]
    elif args.valid_ratio == 0:
        stat = convert(data_map, shuffled_list, args.train_json_file, categories, 'train')
        irre_data_cnt += stat[0]
        bad_image_data += stat[1]
        invalid_data_cnt += stat[2]
    else:
        stat = convert(data_map, shuffled_list[:train_size], args.train_json_file, categories, 'train')
        irre_data_cnt += stat[0]
        bad_image_data += stat[1]
        invalid_data_cnt += stat[2]
        stat = convert(data_map, shuffled_list[train_size:], args.valid_json_file, categories_test, 'valid', args.valid_size_thr)
        irre_data_cnt += stat[0]
        bad_image_data += stat[1]
        invalid_data_cnt += stat[2]
    print('# of files without bbox data entry: ' + str(irre_data_cnt))
    print('# of files with corrupted data: ' + str(bad_image_data))
    print('# of invalid files: ' + str(invalid_data_cnt))