import argparse
import json
import os
from PIL import Image
import shutil
import sys

from glob import glob
from os.path import join as pj
from PIL import Image

from sklearn.model_selection import train_test_split

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, pj(cur_path, '..', '..', '..', '..'))
from yx_toolset.python.utils.data_conversion import parse_folder_ft_det

def merge_cls(data_path, target_cls_list):
    """
        Args:
            data_path: root dir of the data
            target_cls: class to merge to
    """
    for target_cls in target_cls_list:
        folder_list = [path for path in os.listdir(data_path) if path.startswith(target_cls) and path != target_cls]
        file_list = []
        for folder in folder_list:
            file_list += glob(pj(data_path, folder, '*'))
        for file in file_list:
            shutil.move(file, pj(data_path, target_cls))
        for folder in folder_list:
            os.rmdir(pj(data_path, folder))

def set_split(data_path, testset_path, test_ratio=0.2):
    file_paths = []
    file_labels = []
    cat_list = [os.path.basename(path) for path in glob(pj(data_path, '*'))]
    for cat in cat_list:
        for file_path in glob(pj(data_path, cat, '*')):
            file_paths.append(file_path)
            file_labels.append(cat)

    test_size = int(len(file_paths) * test_ratio)
    # print(test_size)
    (trainPaths, testPaths, trainLabels, testLabels) = train_test_split(file_paths, 
                                                                        file_labels, 
                                                                        test_size=test_size, 
                                                                        stratify=file_labels, 
                                                                        random_state=42)
    for f_path, f_label in zip(testPaths, testLabels):
        target_dir = pj(testset_path, f_label)
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        shutil.move(f_path, target_dir)

def size_filter(classify_folder_path):
    """
        Args: 
            classify_folder_path: the directory that contains the folder of each class

    """
    # TODO: Add warning for this operation
    # Warning: Backup your data b4 using this
    img_path_list = []
    for cat in os.listdir(classify_folder_path):
        img_path_list += glob(pj(classify_folder_path, cat, '*'))
    for img_path in img_path_list:
        img = Image.open(img_path)
        shorter_side = min(img.size[0], img.size[1])
        if shorter_side < 25:
            os.remove(img_path)

def cls_check(cls_name):
    if cls_name.startswith('p') and cls_name != 'panel':
        return True

def extract_clsf_data_from_od_data(input_path, extend_scale, output_path):
    data_map, _, _ = parse_folder_ft_det(input_path, False, 0)
    for idz, (k, paths) in enumerate(data_map.items()):
        print('{}/{}'.format(idz + 1, len(data_map)))
        img_path, anno_path = paths[0], paths[1]
        with open(anno_path) as ann_f:
            ann_json = json.load(ann_f)
        if not ann_json['labeled']:
            continue
        pil_im = Image.open(img_path)
        width, height = pil_im.size
        for idx, obj in enumerate(ann_json['outputs']['object']):
            if cls_check(obj['name']):
                try:
                    x1 = int(obj['bndbox']['xmin'])
                    x2 = int(obj['bndbox']['xmax'])
                    y1 = int(obj['bndbox']['ymin'])
                    y2 = int(obj['bndbox']['ymax'])
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    bb_width, bb_height = float(extend_scale) * (x2 - x1), float(extend_scale) * (y2 - y1)
                    new_x1 = max(int(center_x - bb_width / 2), 0)
                    new_x2 = min(int(center_x + bb_width / 2), width)
                    new_y1 = max(int(center_y - bb_height / 2), 0)
                    new_y2 = min(int(center_y + bb_height / 2), height)
                    im_crop = pil_im.crop((new_x1, new_y1, new_x2, new_y2))
                    if not os.path.isdir(pj(output_path, obj['name'])):
                        os.makedirs(pj(output_path, obj['name']))
                    im_crop.save(pj(output_path, obj['name'], os.path.basename(img_path).replace('.', '_' + str(idx) + '.')))
                except:
                    print(obj)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('function', choices=['patch_extract', 'class_filter', 'set_split'])
    parser.add_argument('--input_path', 
                        default=['/media/yingges/Data_Junior/data/ft_pic/GIS-SIGN_optimize/annotation-20191121/device_didi',
                                 '/media/yingges/Data_Junior/data/ft_pic/GIS-SIGN_arrows/arrows-20191107',
                                 '/media/yingges/Data_Junior/data/ft_pic/GIS-SIGN_02/annotation-20191205',
                                 '/media/yingges/Data_Junior/data/ft_pic/GIS-SIGN_02/online-20191128',
                                 '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup'])
    parser.add_argument('--output_path', default='/media/yingges/Data_Junior/data/classification/Padded_2')
    parser.add_argument('--extend_scale', default=2, type=float)
    return parser.parse_args()

if __name__ == '__main__':
    # for cat in ['ph', 'pl', 'pm']:
    #   merge_cls('/media/yingges/Data_Junior/data/crop', cat)
    # set_split('/media/yingges/Data_Junior/data/classification/crop', '/media/yingges/Data_Junior/data/classification/20200107/test')
    # size_filter('/media/yingges/Data_Junior/data/classification/20200107_test_size_thres/val')

    # size_filter('/media/yingges/Data_Junior/data/classification/20200107_all_size_thres/train')
    # size_filter('/media/yingges/Data_Junior/data/classification/20200107_all_size_thres/val')

    args = parse_args()

    if args.function == 'patch_extract':
        extract_clsf_data_from_od_data(args.input_path, args.extend_scale, args.output_path)
    elif args.function == 'class_filter':
        merge_cls('/media/yingges/Data_Junior/data/classification/Padded_2',
                  ['ph', 'pl', 'pm'])
    elif args.function == 'set_split':
        set_split(args.input_path, args.output_path)