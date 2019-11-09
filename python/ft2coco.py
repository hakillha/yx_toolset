import json
import os
from os.path import join as pj

# PRE_DEFINE_CATEGORIES = None
PRE_DEFINE_CATEGORIES = {'roa': 1, 'loa': 2, 'soa': 3, 'sloa': 4, 'sroa': 5,
                         'ooa': 6, 'cf': 7, 'rg': 8, 'np': 9, 'cross': 10}

def get_categories(ft_files):
    classes_names = set()
    for filename in ft_files:
        file = open(filename)
        json_dict = json.load(file)
        for top_tuple in json_dict.keys():

            if top_tuple == 'outputs':
                # print out the images that don't have annotation
                # if 'object' not in json_dict['outputs'].keys():
                #     print(filename + ' contains no object?')
                for output in json_dict['outputs'].keys():

                    if output == 'object':
                        for bb in json_dict['outputs']['object']:
                            classes_names.add(bb['name'])
            
    return {name: i for i, name in enumerate(classes_names)}

def write_one_image(json_dict, out_json_dict, categories, img_id, anno_id):
    # Check if this image contains class of interest. If not, skip it
    contain_coi = False
    for anno in json_dict['outputs']['object']:
        if anno['name'] in categories:
            if 'polygon' in anno.keys(): # rule out the bb annotation
                contain_coi = True
    if not contain_coi:
        return

    img_fname = os.path.basename(json_dict['path'].replace('\\', '/'))
    width = json_dict['size']['width']
    height = json_dict['size']['height']
    img_info = {
        'file_name': img_fname,
        'height': height,
        'width': width,
        'id': img_id
    }
    out_json_dict['images'].append(img_info)
    

    # paste this whenever you want to visualize the json record
    # print(json.dumps(json_dict, indent=4))
    for anno in json_dict['outputs']['object']:
        if anno['name'] in categories.keys() and 'polygon' in anno.keys():
            cate_id = categories[anno['name']]
            point_set = []
            for i in range(len(anno['polygon']) / 2):
                point_set.append(anno['polygon']['x' + str(i + 1)])
                point_set.append(anno['polygon']['y' + str(i + 1)])
            anno_entry = {
                'iscrowd': 0,
                'image_id': img_id,
                'category_id': cate_id,
                'id': anno_id,
                'ignore': 0,
                'segmentation': point_set,
            }
            out_json_dict['annotations'].append(anno_entry)
            anno_id += 1   
    img_id += 1
    return img_id, anno_id

def convert(ft_files, json_file):
    # check for filename in the image folder?
    out_json_dict = {'images': [], 'type': 'instances', 'annotations': [], 'categories': []}
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(ft_files)

    img_id = 1
    anno_id = 1
    for fname in ft_files:
        file = open(fname)
        json_dict = json.load(file)
        for top_tuple in json_dict.keys():

            if top_tuple == 'outputs':
                for output in json_dict['outputs'].keys():

                    # This is equivalent to checking json['labeled'] but is more generalized
                    if output == 'object':
                        ret = write_one_image(json_dict, out_json_dict, categories, img_id, anno_id)
                        if ret:
                            img_id, anno_id = ret

    for cate, cid in categories.items():
        cate_entry = {'supercategory': 'none', 'id': cid, 'name': cate}
        out_json_dict['categories'].append(cate_entry)
    # sort it in the out dict
    out_json_dict['categories'].sort(key=lambda val: val['id'])

    # exist_ok is a python3 argument?
    # os.makedirs(os.path.dirname(json_file), exist_ok=True)
    out_file = open(json_file, 'w')
    out_file.write(json.dumps(out_json_dict))
    out_file.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # set as optional for now
    # parser.add_argument('--ft_dir', type=str)
    # parser.add_argument('--coco_json_file', type=str)
    parser.add_argument('--ft_dir', default='yunxikeji-01-2019-10-21', help='Please use absolute path', type=str)
    parser.add_argument('--coco_json_file', default='cocoformat_out.json', type=str)
    args = parser.parse_args()

    # add os.path.dirname(os.path.abspath(__file__)), 
    # or restrict the running dir of this file
    ftfilelist = [pj(args.ft_dir, 'labels', item) for item in os.listdir(pj(args.ft_dir, 'labels')) if item.endswith('.json')]
    convert(ftfilelist, args.coco_json_file)
