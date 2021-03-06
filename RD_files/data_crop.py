import os
import scipy.misc as misc
import json
import numpy as np
import copy, cv2

def save_to_jsons(save_path, im_height, im_width, objects_axis, label_name):
    out_json_dict = {"path": "", "outputs": {"object": []}, "time_labeled": 1574423089041, "labeled": "true",
                     "size": {"width": 0, "height": 0, "depth": 3}}
    out_json_dict["size"]["width"] = im_width
    out_json_dict["size"]["height"] = im_height
    out_json_dict["path"] = save_path
    object_num = len(objects_axis)
    for i in range(object_num):
        object_dict = {"name": "", "bndbox": {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}}
        object_dict["name"] = label_name[int(objects_axis[i][-1])]
        object_dict["bndbox"]["xmin"], object_dict["bndbox"]["ymin"], object_dict["bndbox"]["xmax"], object_dict["bndbox"]["ymax"] \
            = str((objects_axis[i][0])), str((objects_axis[i][1])), str((objects_axis[i][2])), str((objects_axis[i][3]))
        out_json_dict["outputs"]["object"].append(object_dict)
    #print(out_json_dict)
    out_file = open(save_path, 'w')
    out_file.write(json.dumps(out_json_dict))
    out_file.close()

def clip_image(file_idx, image, boxes_all, width, height):
    if len(boxes_all) > 0:
        shape = image.shape
        for start_h in range(0, shape[0], 450):
            for start_w in range(0, shape[1], 450):
                boxes = copy.deepcopy(boxes_all)
                box = np.zeros_like(boxes_all)
                start_h_new = start_h
                start_w_new = start_w
                if start_h + height > shape[0]:
                    start_h_new = shape[0] - height
                if start_w + width > shape[1]:
                    start_w_new = shape[1] - width
                top_left_row = max(start_h_new, 0)
                top_left_col = max(start_w_new, 0)
                bottom_right_row = min(start_h + height, shape[0])
                bottom_right_col = min(start_w + width, shape[1])

                subImage = image[top_left_row:bottom_right_row, top_left_col: bottom_right_col]
                box[:, 0] = boxes[:, 0] - top_left_col
                box[:, 2] = boxes[:, 2] - top_left_col

                box[:, 1] = boxes[:, 1] - top_left_row
                box[:, 3] = boxes[:, 3] - top_left_row

                box[:, 4] = boxes[:, 4]
                center_y = 0.5 * (box[:, 1] + box[:, 3] )
                center_x = 0.5 * (box[:, 0] + box[:, 2] )

                cond1 = np.intersect1d(np.where(center_y[:] >= 0)[0], np.where(center_x[:] >= 0)[0])
                cond2 = np.intersect1d(np.where(center_y[:] <= (bottom_right_row - top_left_row))[0],
                                       np.where(center_x[:] <= (bottom_right_col - top_left_col))[0])
                idx = np.intersect1d(cond1, cond2)
                #print(idx)
                if len(idx) > 0:
                    jsons = os.path.join(save_dir,
                                       "%s_%04d_%04d.json" % (file_idx, top_left_row, top_left_col))
                    save_to_jsons(jsons, subImage.shape[0], subImage.shape[1], box[idx, :], class_list)
                    #print ('save json : ', jsons)
                    if subImage.shape[0] > 5 and subImage.shape[1] > 5:
                        img = os.path.join(save_dir, 'images',
                                           "%s_%04d_%04d.jpg" % (file_idx, top_left_row, top_left_col))
                        cv2.imwrite(img, subImage)



class_list = ['io', 'wo', 'ors', 'p10', 'p11', 
                         'p26', 'p20', 'p23', 'p19', 'pne', 
                         'rn', 'ps', 'p5', 'lo', 'tl', 
                         'pg',  'ro', 'pn','ph1.9','pw1.85','pm49','sc','pa','pw','pl30r','pl11','pr5','ph2r','pm3','ph2,8','pr15','il60 ','po20',
                         'po', 'pl', 'pm','bak','ph1','pa20','ph1.55','ph','sc0',"sc1","ors","i1", "i10", "i11", "i12", "i13", "i14", "i15", "i2", "i3", "i4",
              "i5", "il100", "il110", "il50", "il60", "il70", "il80", "il90", "io", "ip", "p1", "p10", "p11", "p12", "p13",
              "p14", "p15", "p16", "p17", "p18", "p19", "p2", "p20", "p21", "p22", "p23", "p24", "p25", "p26", "p27", "p28", "p3",
              "p4", "p5", "p6", "p7", "p8", "p9", "pa10", "pa12", "pa13", "pa14", "pa8", "pb", "pc", "pg", "ph1.5", "ph2", "ph2.1",
              "ph2.2", "ph2.4", "ph2.5", "ph2.8", "ph2.9", "ph3", "ph3.2", "ph3.5", "ph3.8", "ph4", "ph4.2", "ph4.3", "ph4.5", "ph4.8",
              "ph5", "ph5.3", "ph5.5", "pl10", "pl100", "pl110", "pl120", "pl15", "pl20", "pl25", "pl30", "pl35", "pl40", "pl5", "pl50",
              "pl60", "pl65", "pl70", "pl80", "pl90", "pm10", "pm13", "pm15", "pm1.5", "pm2", "pm20", "pm25", "pm30", "pm35", "pm40", "pm46",
              "pm5", "pm50", "pm55", "pm8", "pn", "pne", "po", "pr10", "pr100", "pr20", "pr30", "pr40", "pr45", "pr50", "pr60", "pr70", "pr80", "ps", "pw2", "pw2.5", "pw3", "pw3.2", "pw3.5", "pw4",
              "pw4.2", "pw4.5", "w1", "w10", "w12", "w13", "w16", "w18", "w20", "w21", "w22", "w24", "w28", "w3", "w30", "w31", "w32", "w34", "w35", "w37", "w38", "w41", "w42", "w43", "w44",
              "w45", "w46", "w47", "w48", "w49", "w5", "w50", "w55", "w56", "w57", "w58", "w59", "w60", "w62", "w63", "w66", "w8", "wo", "i6", "i7", "i8", "i9", "ilx", "p29", "w29", "w33",
              "w36", "w39", "w4", "w40", "w51", "w52", "w53", "w54", "w6", "w61", "w64", "w65", "w67", "w7", "w9", "pax", "pd", "pe", "phx", "plx", "pmx", "pnl", "prx", "pwx", "w11",
              "w14", "w15", "w17", "w19", "w2", "w23", "w25", "w26", "w27", "pl0", "pl4", "pl3", "pm2.5", "ph4.4", "pn40", "ph3.3", "ph2.6"]

'''class_list = ['io', 'wo', 'ors', 'p10', 'p11', 
                         'p26', 'p20', 'p23', 'p19', 'pne', 
                         'rn', 'ps', 'p5', 'lo', 'tl', 
                         'pg',  'ro', 'pn', 
                         'po', 'pl', 'pm']
'''

raw_label_dir = '/home/yingges/Downloads/tt100kjson/'
raw_images_dir = '/home/yingges/Downloads/tt100kjpg/'
save_dir = '/home/yingges/Downloads/crop/'


images = [i for i in os.listdir(raw_images_dir) if 'jpg' in i]
labels = [i for i in os.listdir(raw_label_dir) if 'json' in i]

print('find image', len(images))
print('find label', len(labels))

min_length = 1e10
max_length = 1

for idx, jsons in enumerate(labels):
    print(idx, 'read json', jsons)
    #print(os.path.join(raw_label_dir, jsons))
    json_file = json.loads(open(os.path.join(raw_label_dir, jsons)).read())
    outputs = json_file["outputs"]["object"]
    box = []
    for object_ in outputs:
        #print(object_)
        if object_["name"] not in class_list:
            print('warning found a new label :', object_["name"])
            #continue
            exit()
        box.append([int(object_["bndbox"]["xmin"]), int(object_["bndbox"]["ymin"]), int(object_["bndbox"]["xmax"]), int(object_["bndbox"]["ymax"])]+ [class_list.index(object_["name"])])

    img_data = cv2.imread(os.path.join(raw_images_dir, jsons.replace('json', 'jpg')))

    clip_image(jsons.strip('.json'), img_data, np.array(box), 600, 1000)
