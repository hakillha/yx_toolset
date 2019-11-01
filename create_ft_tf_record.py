import argparse
import hashlib
import io
import json
import os
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
# Patch the location of gfile
tf.gfile = tf.io.gfile
from os.path import join as pj

BAD_DATA = 0
ALL_DATA = 0

def find_parent_class(child_cls):
    if child_cls in ['ph1','ph1.55','ph1.6','ph1.9','ph2',
                     'ph2,8','ph2.1','ph2.2','ph2.4','ph2.5',
                     'ph2.8','ph2r','ph3','ph3.2','ph3.5',
                     'ph3.6','ph4','ph4.2','ph4.5','ph4.8',
                     'ph5','ph5.5']:
        return 'ph'
    if child_cls in ['pl10','pl11','pl14','pl20','pl3',
                     'pl30','pl30r','pl35','pl40','pl5',
                     'pl50','pl60','pl70','pl80']:
        return 'pl'
    if child_cls in ['pm10','pm20','pm3','pm30','pm40',
                     'pm49','pm5','pm50','pm55','pm8']:
        return 'pm'
    return child_cls

def ft_to_tf_example(ft_label_file_name,
                     ft_dir,
                     local_fname,
                     label_map_dict):
    global BAD_DATA
    label_file = open(ft_label_file_name, 'r')
    json_dict = json.load(label_file)
    if 'object' in json_dict['outputs'].keys():
        # check existence of coi
        contain_coi = False
        for obj in json_dict['outputs']['object']:
            if find_parent_class(obj['name']) in label_map_dict.keys():
                contain_coi = True
        if not contain_coi:
            return

    # Comment out to speed up debug runs
    img_path = pj(ft_dir, 'images', local_fname.split('.')[0] + '.jpg')
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    # check image format
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)

    if not image.verify():
        BAD_DATA += 1
        return

    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(json_dict['size']['width'])
    height = int(json_dict['size']['height'])

    width_og, height_og = image.size
    # if width_og != width or height_og != height:
    #     print('Inconsistent data: '+ str(width) + ', ' +str(height) + '; ' +
    #           str(width_og) + ', ' +str(height_og))
    #     BAD_DATA += 1
    #     width = width_og
    #     height = height_og

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    print(ft_label_file_name)
    if 'object' in json_dict['outputs'].keys():
        for obj in json_dict['outputs']['object']:
            bb_class = find_parent_class(obj['name'])
            if bb_class not in label_map_dict.keys():
                continue
            difficult_obj.append(0)

            test01 = float(obj['bndbox']['xmin']) / width
            test02 = float(obj['bndbox']['ymin']) / height
            test03 = float(obj['bndbox']['xmax']) / width
            test04 = float(obj['bndbox']['ymax']) / height
            # print(test01)
            # assert test01 <= 1
            # print(test02)
            # assert test02 <= 1
            # print(test03)
            # assert test03 <= 1
            # print(test04)
            # assert test04 <= 1
            if test01 > 1 or test02 > 1 or test03 > 1 or test04 > 1:
                print('Inconsistent data: '+ str(width) + ', ' +str(height) + '; ' +
                      str(width_og) + ', ' +str(height_og))


            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(bb_class.encode('utf8'))
            classes.append(label_map_dict[bb_class])
            truncated.append(0)
            poses.append('Unspecified'.encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          os.path.basename(img_path).encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          os.path.basename(img_path).encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
      }))
    return example

def main(_):
    parser = argparse.ArgumentParser()
    # set as optional for now
    # better require abs path
    parser.add_argument('--output_path', default='ft_tfod.record')
    parser.add_argument('--label_map_path', default='ft_od_label_map.txt')
    parser.add_argument('--ft_data_dir', help='Should be absolute path', default='../yunxikeji_01_label-20190927')
    args = parser.parse_args()

    writer = tf.io.TFRecordWriter(args.output_path)

    label_map_dict = label_map_util.get_label_map_dict(args.label_map_path)

    data_dir = [item for item in os.listdir(args.ft_data_dir) if os.path.isdir(pj(args.ft_data_dir, item))]
    for folder in data_dir:
        label_list = pj(args.ft_data_dir, folder, 'labels')
        if os.path.exists(label_list): # check if it's a valid data folder
            for label_fname in os.listdir(label_list):
                global ALL_DATA
                ALL_DATA += 1
                tf_example = ft_to_tf_example(pj(label_list, label_fname),
                                              pj(args.ft_data_dir, folder),
                                              label_fname,
                                              label_map_dict)
                if tf_example:
                    writer.write(tf_example.SerializeToString())
    writer.close()

    print(ALL_DATA)
    print(BAD_DATA)

if __name__ == '__main__':
    tf.compat.v1.app.run()