import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import skimage.io as io
import sys
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from os.path import join as pj
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate(annFile, resFile, annType, per_cls_stat=False):
    """
        Args: annType has the following types ['segm','bbox','keypoints']

    """
    res = []
    cocoGt=COCO(annFile)
    cocoDt=cocoGt.loadRes(resFile)
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    res.append(cocoEval.stats[1])
    
    catIds = cocoGt.getCatIds()
    cats = cocoGt.loadCats(catIds)
    for cat in cats:
        print('\nClass ' + cat['name'] + ' stats:')
        cocoEval.params.catIds = [cat['id']]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        res.append(cocoEval.stats[1])
    cats = [{'name': 'Overall'}] + cats
    for idx, cls_ in enumerate(cats):
        print('{} mAP@.5: {:0.3f}'.format(cats[idx]['name'], res[idx]))

def showBndbox(coco, anns):
    """
        Args: Dataset object in coco format

    """
    ax = plt.gca()
    rectangles = []
    color = []
    for ann in anns:
        c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        if 'bbox' in ann:
            # score = float(ann['score'])
            # if score < 0.5: continue
            bb = ann['bbox']
            rectangles.append(Rectangle((bb[0], bb[1]), bb[2], bb[3]))
            color.append(c)
            text = coco.loadCats(ann['category_id'])[0]['name']
            # if 'score' in ann:
            #   text = '%s: %.2f' % (coco.loadCats(ann['category_id'])[0]['name'], score)
            # else:
            #   text = coco.loadCats(ann['category_id'])[0]['name']
            ax.text(bb[0], bb[1] - 15, text, color=[1, 1, 1], backgroundcolor=c, weight='bold')
    p = PatchCollection(rectangles, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)

def coco_format_viz(img_folder, annFile):
    coco = COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print(' '.join(nms))
    # catIds = coco.getCatIds(catNms=[nms[np.random.randint(0, len(nms))]])
    # imgIds = coco.getImgIds(catIds=catIds)
    # img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    # I = io.imread(pj(img_folder, img['file_name']))

    # plt.imshow(I); plt.axis('off')
    # annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    # anns = coco.loadAnns(annIds)
    # # coco.showAnns(anns)
    # showBndbox(coco, anns)
    # plt.show()

    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()

    for imgId in imgIds:
        img = coco.loadImgs(imgId)[0]
        I = io.imread(pj(img_folder, img['file_name']))

        plt.imshow(I); plt.axis('off')
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        showBndbox(coco, anns)
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mode', default='eval')
    parser.add_argument('--ann_type', default='bbox')
    parser.add_argument('--anno_file_path', default='/media/yingges/Data/Datasets/COCO/annotations/instances_val2017.json', type=str)
    parser.add_argument('--img_folder_path', default='/media/yingges/Data/Datasets/COCO/val2017',type=str)
    parser.add_argument('--res_file_path', default='/media/yingges/Data/201910/yolact/yolact/results/mask_detections.json')
    parser.add_argument('--per_cls_stat', action='store_true')

    args = parser.parse_args()

    # args.anno_file_path = '/media/yingges/Data/201910/FT/FTData/ft_od1_merged/other/sample100/sample_ann.json'
    # args.res_file_path = '/media/yingges/Data/201910/FT/FTData/ft_od1_merged/other/sample100/map_output.json'
    # args.anno_file_path = '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/cocoformat_valid_out.json'
    # args.res_file_path = '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/map_output_1130.json'
    args.anno_file_path = '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/ignore_toosmall/11_30/valid.json'
    args.res_file_path = '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/ignore_toosmall/11_30/epoch21_output.json'

    if args.mode == 'viz':
        coco_format_viz(args.img_folder_path, args.anno_file_path)
    if args.mode == 'eval':
        evaluate(args.anno_file_path, args.res_file_path, args.ann_type, args.per_cls_stat)

if __name__ == '__main__':
    main()