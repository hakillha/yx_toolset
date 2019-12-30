import argparse
import json

from collections import defaultdict
from pycocotools.coco import COCO

import post_proc_cfg

def cls_filter(cat_name, filter_rule):
    # add a new class selection routine by adding a new
    # "if filter_rule == '...':" branch and a new cfg dict in 
    # the 'post_proc_cfg' file
    if filter_rule == 'exclude_p':
        cfg = post_proc_cfg.exclude_p
        if cat_name in cfg['irre_cls_list'] or (cat_name.startswith('p') and cat_name != 'panel'):
            return None
        if cfg['strip_space']:
            cat_name = cat_name.strip()
        # the following lines would also remove the space
        # so be careful with the logic flow here
        if cat_name in cfg['panel_cls_list']:
            cat_name = 'panel'

        if cat_name in cfg['i_cls_include_list']:
            cat_name = 'i'
        if cat_name in cfg['w_cls_include_list']:
            cat_name = 'w'

    if filter_rule == 'p_finegrained':
        cfg = post_proc_cfg.p_finegrained
        if cfg['strip_space']:
            cat_name = cat_name.strip()
        if not cat_name.startswith('p') or cat_name == 'panel':
            return None
        if cfg['ignore_number']:
            for pre in cfg['sign_merge_list']:
                # if we want to keep the spaces
                # if cat_name.startswith(pre) and cat_name !=  pre + ' ':
                if cat_name.startswith(pre):
                    cat_name = pre
        if cat_name not in cfg['p_cls_include_list']:
            return None
    return cat_name

def anns_post_proc(ann_list, filter_rule):
    new_ann_list = []
    for ann in ann_list:
        cat_name = cls_filter(ann['category_name'], filter_rule)
        if cat_name == None:
            continue
        ann['category_name'] = cat_name
        new_ann_list.append(ann)
    return new_ann_list

def ann_stats(ann_list):
    class_list = set()
    class_cnt = defaultdict(int)
    img_id_list = set()
    for ann in ann_list:
        class_list.add(ann['category_name'])
        class_cnt[ann['category_name']] += 1
        img_id_list.add(ann['image_id'])
    class_list = list(class_list)
    class_list.sort()
    print(class_list)
    print(class_cnt)
    return class_list, class_cnt, img_id_list

def post_proc(input_coco_file, filter_rule, train_coco_file, test_coco_file, test_ratio, eval_stats):
    train_json_dict = {'images': [], 'type': 'instances', 'annotations':[], 'categories':[]}
    test_json_dict = {'images': [], 'type': 'instances', 'annotations':[], 'categories':[]}
    in_coco_ds = COCO(input_coco_file)
    with open(input_coco_file) as file:
        json_dict = json.load(file)
    new_ann_list = anns_post_proc(json_dict['annotations'], filter_rule)
    # After filtering the anns:
    # 1. Iterate the ann list, regenerate the categories map
    # 2. Iterate through the ann list again, fix the cat id based on the new cats map
    # and collect the image ids that should be kept.
    # 3. Maybe double check the cat map with a given one
    # 4. Generate a train/test split and collect the corresponding 
    # anns based on the images split
    class_list, _, img_id_list = ann_stats(new_ann_list)
    categories = {}
    for idx, cat in enumerate(class_list):
        cat_entry = {'supercategory': 'none', 'id': idx + 1, 'name': cat}
        categories[cat] = idx + 1
        train_json_dict['categories'].append(cat_entry)
        test_json_dict['categories'].append(cat_entry)
    for ann in new_ann_list:
        ann['category_id'] = categories[ann['category_name']]
    img_id_list = list(img_id_list)
    num_train = int(len(img_id_list) * (1.0 - test_ratio))
    train_img_info = in_coco_ds.loadImgs(img_id_list[:num_train])
    test_img_info = in_coco_ds.loadImgs(img_id_list[num_train:])
    # update ann field in coco datasets so we can extract them with coco interface
    in_coco_ds.dataset['annotations'] = new_ann_list
    in_coco_ds.createIndex()
    train_anns = in_coco_ds.loadAnns(in_coco_ds.getAnnIds(imgIds=img_id_list[:num_train]))
    test_anns = in_coco_ds.loadAnns(in_coco_ds.getAnnIds(imgIds=img_id_list[num_train:]))
    # ann_stats(train_anns)
    # ann_stats(test_anns)
    # if eval_stats:
    #     return
    train_json_dict['images'] = train_img_info
    test_json_dict['images'] = test_img_info
    train_json_dict['annotations'] = train_anns
    test_json_dict['annotations'] = test_anns
    with open(train_coco_file, 'w') as train_file:
        train_file.write(json.dumps(train_json_dict))
    with open(test_coco_file, 'w') as test_file:
        test_file.write(json.dumps(test_json_dict))

def parse_args():
    parser = argparse.ArgumentParser("""""")
    # change to required arguments if released
    parser.add_argument('--input_coco_file')
    parser.add_argument('--train_coco_file')
    parser.add_argument('--test_coco_file')
    parser.add_argument('--filter_rule')
    parser.add_argument('--test_ratio', default=0.25)
    parser.add_argument('--eval_stats')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    args.input_coco_file = '/media/yingges/Data_Junior/data/ft_pic/super_ensemble.json'
    args.train_coco_file = '/media/yingges/Data_Junior/data/ft_pic/p_finegrained/train.json'
    args.test_coco_file = '/media/yingges/Data_Junior/data/ft_pic/p_finegrained/test.json'
    # args.filter_rule = 'exclude_p'
    args.filter_rule = 'p_finegrained'

    post_proc(args.input_coco_file,
              args.filter_rule,
              args.train_coco_file,
              args.test_coco_file,
              args.test_ratio,
              args.eval_stats)