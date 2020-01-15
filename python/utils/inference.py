import json

from os.path import join as pj

def generate_eval_img_info(img_folder, test_json_file):
    imgs = []
    img_ids = []
    with open(test_json_file) as file:
        test_json = json.load(file)
    for img_info in test_json['images']:
        imgs.append(pj(img_folder, img_info['file_name']))
        img_ids.append(img_info['id'])
    return imgs, img_ids

def generate_result_record(infer_res, img_id, cat_map):
    img_res_list = []
    for res_idx in range(len(infer_res['cls_name'])):
        img_res_list.append({'image_id': img_id, 
                             'category_id': cat_map[infer_res['cls_name'][res_idx]], 
                             'bbox': [float(infer_res['bbox'][res_idx][0]),
                                      float(infer_res['bbox'][res_idx][1]),
                                      float(infer_res['bbox'][res_idx][2] - infer_res['bbox'][res_idx][0]),
                                      float(infer_res['bbox'][res_idx][3] - infer_res['bbox'][res_idx][1])], 
                             "score": float(infer_res['score'][res_idx])})
    return img_res_list