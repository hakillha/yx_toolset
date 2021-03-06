import argparse
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import random
import skimage.io as io
import sys
import pylab
# pylab.rcParams['figure.figsize'] = (12.0, 15.0)

from collections import defaultdict
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from os.path import join as pj
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from matplotlib.patches import Rectangle, Polygon

def evaluate(annFile, resFile, annType):
    """
        Args: annType has the following types ['segm','bbox','keypoints']

    """
    res = []
    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)
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

def evaluate_curve(annFile, resFile, annType, score_thr, segm_iou_thr, split=10):
    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)
    anns = cocoDt.dataset['annotations']
    anns = [ann for ann in anns if ann['score'] >= score_thr]

    cocoDt.dataset['annotations'] = anns
    cocoDt.createIndex()
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    if annType == 'segm':
        cocoEval.params.iouThrs = np.linspace(segm_iou_thr, 0.95, np.round((0.95 - .3) / .05) + 1, endpoint=True)
    cocoEval.evaluate()
    # maxDet = -1 # turn off maxDet
    res_summary = []
    maxDet = 1000
    K0 = len(cocoEval.params.catIds)
    A0 = len(cocoEval._paramsEval.areaRng)
    I0 = len(cocoEval._paramsEval.imgIds)
    for k in range(K0):
        E = [cocoEval.evalImgs[k * A0 * I0 + a * I0 + i] 
                        for i in range(I0)
                        for a in [0]]
        E = [e for e in E if not e is None]
        dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])
        inds = np.argsort(-dtScores, kind='mergesort')
        dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
        dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
        gtIg = np.concatenate([e['gtIgnore'] for e in E])
        npig = np.count_nonzero(gtIg==0)

        tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
        fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )
        tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
        fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
        if annType == 'bbox':
            tp_sum_50 = np.array(tp_sum[0])
            fp_sum_50 = np.array(fp_sum[0])
            rc = tp_sum_50 / npig
            pr = tp_sum_50 / (fp_sum_50+tp_sum_50+np.spacing(1))
        elif annType == 'segm':
            """Followings are double IOU thresholds code"""
            # tp_sum_40 = np.array(tp_sum[2])
            # fp_sum_40 = np.array(fp_sum[2])
            # # pr_iou_40
            # pr = tp_sum_40 / (fp_sum_40+tp_sum_40+np.spacing(1))
            # tp_sum_30 = np.array(tp_sum[0])
            # # rc_iou_30
            # rc = tp_sum_30 / npig

            tp_sum = np.array(tp_sum[0])
            fp_sum = np.array(fp_sum[0])
            rc = tp_sum / npig
            pr = tp_sum / (fp_sum+tp_sum+np.spacing(1))
        print(cocoGt.loadCats(k + 1)[0]['name'])
        for i in range(0, len(rc), int(len(rc) / split)):
            print('ACC: {:.3f}, RECALL: {:.3f}'.format(pr[i], rc[i]))
        res_summary.append((pr[-1], rc[-1]))
    for idx, res in enumerate(res_summary):
        print(cocoGt.loadCats(idx + 1)[0]['name'] + ': {:.3f}, {:.3f}'.format(res[0], res[1]))

    # cocoDt = cocoGt.loadRes(anns)
    # cocoEval = COCOeval(cocoGt,cocoDt,annType)
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()
    # for i in range(0, 100, 2):
    #     pr = np.mean(cocoEval.eval['precision'][0,i,:,0,2])
    #     print('ACC: {:.3f}, RECALL: {:.3f}'.format(pr, (i + 1) / 100.0))
    # prs = [np.mean(cocoEval.eval['precision'][0,i,:,0,2]) for i in range(0, 101)]
    # x = np.arange(0.0, 1.01, 0.01)
    # plt.xlabel('recall')
    # plt.ylabel('precision')
    # plt.xlim(0, 1.0)
    # plt.ylim(0, 1.01)
    # plt.grid(True)
    # plt.plot(x, prs)
    # plt.show()

    # anns = sorted(anns, key=lambda ann: -ann['score'])
    # n = int(len(anns) / split) # chunk length
    # anns_chunks = [anns[0:i + n] for i in range(0, len(anns), n)]
    # num_gt = len(cocoGt.anns)
    # output = []
    # for chunk in anns_chunks:
    #     cocoDt.dataset['annotations'] = chunk
    #     cocoDt.createIndex()
    #     cocoEval = COCOeval(cocoGt,cocoDt,annType)
    #     cocoEval.evaluate()
    #     num_hit = 0
    #     K0 = len(cocoEval.params.catIds)
    #     A0 = len(cocoEval._paramsEval.areaRng)
    #     I0 = len(cocoEval._paramsEval.imgIds)
    #     res_list = [cocoEval.evalImgs[k * A0 * I0 + a * I0 + i] 
    #                 for i in range(0, I0)
    #                 # for a in [0] 
    #                 for a in [1, 2, 3] 
    #                 for k in range(0, K0)]
    #     res_list = [res for res in res_list if not res is None]
    #     # use dtScores to mask off different thres
    #     for res in res_list:
    #         num_hit += np.count_nonzero(np.logical_and(res['gtMatches'][0], np.logical_not(res['gtIgnore']))) # .5 IOU hits
    #     output.append((num_hit / len(chunk), num_hit / num_gt))
    # for out in output:
    #     print('ACC: {:.3f}, RECALL: {:.3f}'.format(out[0], out[1]))

def showBndbox(cats, anns, predefined_c=None):
    """
        Args: Dataset object in coco format

    """
    ax = plt.gca()
    rectangles = []
    color = []
    for ann in anns:
        if 'bbox' in ann:
            bb = ann['bbox']
            rectangles.append(Rectangle((bb[0], bb[1]), bb[2], bb[3]))
            if predefined_c is None:
                c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
            else:
                c = predefined_c
            color.append(c)
            text = cats[ann['category_id'] - 1]['name']
            if predefined_c == [1, 0, 0]:
                ax.text(bb[0] + bb[2], bb[1] - 15, text, color=[1, 1, 1], backgroundcolor=c, weight='bold')
            else:
                ax.text(bb[0], bb[1] - 15, text, color=[1, 1, 1], backgroundcolor=c, weight='bold')
    p = PatchCollection(rectangles, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)

def showPlg(cats, res, gt=None, display_bb=False, textonbox=True):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    rectangles = []
    color = []
    for out in res:
        cat_id = out['category_id']
        score = out['score']
        box = out['bbox']
        mask = maskUtils.decode(out['segmentation'])
        if score < 0.5: continue
        if gt is not None:
            c = (1,0,0)
        else:
            c = np.random.random((1, 3)).tolist()[0]
        img = np.ones((mask.shape[0], mask.shape[1], 3))
        for i in range(3):
            img[:,:,i] = c[i]
        ax.imshow(np.dstack((img, mask * 0.5)))
        rectangles.append(Rectangle((box[0], box[1]), box[2], box[3]))
        color.append(c)
        if textonbox:
            textp1, textp2 = box[0], box[1] - 15
        else:
            textp1, textp2 = np.where(mask)[0][1], np.where(mask)[0][0]
        ax.text(textp1, textp2, cats[cat_id-1]['name'], color=[1, 1, 1], backgroundcolor=c, weight='bold')
    if gt is not None:
        polygons = []
        color = []
        for ann in gt:
            seg = ann['segmentation'][0]
            c_gt = (0,1,0)
            poly = np.array(seg).reshape((int(len(seg)/2), 2))
            polygons.append(Polygon(poly))
            color.append(c_gt)
            ax.text(seg[0], seg[1], cats[ann['category_id'] - 1]['name'], color=[1, 1, 1], backgroundcolor=c_gt, weight='bold')
        # p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
        # ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)

    if display_bb:
        p = PatchCollection(rectangles, facecolor='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)

def coco_format_viz(img_folder, annFile, annType, resFile=None, res_score_thr=0.5, save_path=None):
    assert img_folder is not None, 'Please provide the img folder path!'

    coco = COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    all_cat_id = [cat['id'] for cat in cats]
    catIds = coco.getCatIds(catNms=all_cat_id)
    imgIds = coco.getImgIds(catIds=catIds)

    if resFile is not None:
        with open(resFile) as file:
            json_list = json.load(file)
        res_dict = defaultdict(list)
        for out in json_list:
            if out['score'] >= res_score_thr:
                res_dict[out['image_id']].append(out)

    # TODO: sample imgs
    for i in range(len(imgIds)):
        img = coco.loadImgs(imgIds[i])[0]
        print(img['file_name'])
        I = io.imread(pj(img_folder, img['file_name']))
        plt.imshow(I); plt.axis('off')
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        if annType == 'segm':
            # coco.showAnns(anns)
            showPlg(cats, res_dict[img['id']], anns)
        elif annType == 'bbox':
            if resFile is None:
                showBndbox(cats, anns)
            else:
                showBndbox(cats, anns, [0, 1, 0])
                showBndbox(cats, res_dict[img['id']], [1, 0, 0])
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        if save_path is None:
            plt.show()
        else:
            plt.savefig(pj(save_path, img['file_name']))
        plt.close()

    # catIds = coco.getCatIds()
    # imgIds = coco.getImgIds()
    # # catIds = [0, 1, 20]

    # for imgId in random.sample(imgIds, 50):
    #     img = coco.loadImgs(imgId)[0]
    #     I = io.imread(pj(img_folder, img['file_name']))

    #     plt.imshow(I); plt.axis('off')
    #     annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    #     anns = coco.loadAnns(annIds)
    #     print('Height: {}\n Width: {}\n'.format(img['height'], img['width']))
    #     showBndbox(coco, anns)
    #     fig = plt.gcf()
    #     fig.set_size_inches(18.5, 10.5)
    #     plt.title(img['file_name'])
    #     # plt.savefig(pj('/home/yingges/Desktop/yingges/experiments/data/ft_det_cleanedup/ignore_toosmall/11_30/img_annoed', img['file_name']))
    #     plt.show()
    #     plt.close()

def data_stats(annFile, categories=None):
    with open(annFile) as file:
        json_dict = json.load(file)
    # Collect stats on the shorter side of the bbs
    ann_size_list = []
    cls_cnt = defaultdict(int)
    if categories == None:
        cocoGt = COCO(annFile)
        categories = cocoGt.loadCats(cocoGt.getCatIds())
    print(categories)
    small_cnt = 0
    for ann in json_dict['annotations']:
        cls_name = categories[ann['category_id'] - 1]['name']
        shorter_side = min(ann['bbox'][2], ann['bbox'][3])
        ann_size_list.append(shorter_side)
        if shorter_side < 25:
            small_cnt += 1
        # if cls_name.startswith('p'):
        #     cls_name = 'po'
        cls_cnt[cls_name] += 1
    total_sample_cnt = len(ann_size_list)
    cls_percent = dict()
    for k, v in cls_cnt.items():
        cls_percent[k] = v / total_sample_cnt
    print('Total # of samples: {}'.format(total_sample_cnt))
    print(cls_cnt)
    print(cls_percent)
    counts, edges, plot = plt.hist(ann_size_list, bins=20)
    ann_size_percent = dict()
    for k, v in zip(edges, counts):
        ann_size_percent[k] = v / total_sample_cnt
    print(ann_size_percent)
    print(small_cnt / total_sample_cnt)
    # plt.show()
    plt.close()

    img_size_list = []
    for img in json_dict['images']:
        img_size_list.append(img['height'])
    counts, edges, plot = plt.hist(img_size_list)
    img_size_percent = dict()
    for k, v in zip(edges, counts):
        img_size_percent[k] = v / len(img_size_list)
    print(img_size_percent)
    plt.show()

# PREDEFINED_CLASSES_GENERIC = ['i','p', 'wo', 'rn', 'lo', 'tl',  'ro']
# PREDEFINED_CLASSES_GENERIC_UPDATED_TEST = ['i', 'p', 'wo', 'rn', 'lo', 
#                                            'tl', 'ro', 'sc0', 'sc1']
# PREDEFINED_CLASSES_GENERIC_UPDATED_TRAIN = ['i', 'p', 'wo', 'rn', 'lo', 
#                                             'tl', 'ro', 'sc0', 'sc1', 'ors']
# PREDEFINED_CLASSES = ['io', 'wo', 'ors', 'p10', 'p11', 
#                       'p26', 'p20', 'p23', 'p19', 'pne',
#                       'rn', 'ps', 'p5', 'lo', 'tl',
#                       'pg', 'sc1','sc0', 'ro', 'pn',
#                       'po', 'pl', 'pm']

def main():
    parser = argparse.ArgumentParser(description="""For ease of tracking we store the class_name/class_id list/dictionary in the code.
                                     So please look them up in the code before using this script and change it according to your needs. 
                                     It's particularly important to match the index/id of the classes between the code and the files to 
                                     obtain a correct result.
                                     The above description is no longer valid. The class list metadata should be stored alongside with
                                     gt file now.""")
    parser.add_argument('mode', default='eval', choices=['eval', 'viz', 'stats'],
                                  help="""Behavior of different modes to be added here.
                                  'stats' mode requires [--anno_file_path].
                                  'viz' mode requires [--anno_file_path, --img_folder_path, 
                                                       --res_file_path, --output_save_path(optional)].
                                  'eval' mode requires [--anno_file_path, --res_file_path]""")
    parser.add_argument('--ann_type', default='bbox', choices=['segm','bbox','keypoints'])
    parser.add_argument('--anno_file_path', type=str, 
                                            help='Path to a COCO format annotation json file.')
    parser.add_argument('--img_folder_path',type=str,
                                            help='Path to the folder that holds corresponding images.')
    parser.add_argument('--res_file_path', type=str,
                                           help='Path to a json file that stores model outputs.')
    parser.add_argument('--map_curve', default=False, action='store_true', 
                                       help="""This parameter dictates the behavior of the evaluation process. 
                                       When this is set to true the script will calculate a 10 split global mAP curve and the point
                                       with the lowest confidence threshold would be the result of FT protocol.
                                       When this is left empty it will use standard COCO interfaces to perform COCO evaluation.""")
    parser.add_argument('--score_thr', default=.2, type=float,
                                       help="""This only applies to global average precision calculation. e.g.
                                       When "--map_curve" is specified.""")
    # parser.add_argument('--finegrained_cls', default=False, action='store_true')
    parser.add_argument('--output_save_path', type=str)
    parser.add_argument('--segm_iou_thr', type=float, default=.3)
    args = parser.parse_args()

    # categories = PREDEFINED_CLASSES if args.finegrained_cls else PREDEFINED_CLASSES_GENERIC_UPDATED_TEST
    # categories = PREDEFINED_CLASSES if args.finegrained_cls else PREDEFINED_CLASSES_GENERIC_UPDATED_TRAIN
    if args.mode == 'viz':
        coco_format_viz(args.img_folder_path, 
                        args.anno_file_path, 
                        args.ann_type, 
                        resFile=args.res_file_path, 
                        save_path=args.output_save_path)
    elif args.mode == 'eval':
        if args.map_curve:
            evaluate_curve(args.anno_file_path, args.res_file_path, args.ann_type, args.score_thr, args.segm_iou_thr)
        else:
            evaluate(args.anno_file_path, args.res_file_path, args.ann_type)
        data_stats(args.anno_file_path)
    elif args.mode =='stats':
        data_stats(args.anno_file_path)

if __name__ == '__main__':
    main()