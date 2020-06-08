# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import os
from glob import glob
import json
import time
from tqdm import tqdm




import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box


def display(preds, imgs, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])
        
        if imshow:
            cv2.imshow('_', imgs[i])
            # if cv2.waitKey(0) == 27: break

        if imwrite:
            cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])



if __name__ == "__main__":
    compound_coef = 0
    force_input_size = None  # set None to use default size
    img_path = 'test/img.png'

    # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    obj_list = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    weight = f'weights/efficientdet-d{compound_coef}.pth'
    
    # obj_list= ['person']
    # weight = f'weights/efficientdet-d0_21_339798.pth'

    color_list = standard_to_bgr(STANDARD_COLORS)
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    
    # initialize model
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load(weight))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    
    code = 21
    images = sorted(glob(f'data/hie/images/train/{code:02d}*.jpg'))

    size_thresh = 6


    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for iou_threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            all_anns = []
            ann_id = 0
            for image in tqdm(images):
                # preprocess image
                ori_imgs, framed_imgs, framed_metas = preprocess(image, max_size=input_size)
                # x is stacked as a batch here
                if use_cuda:
                    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
                else:
                    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
                x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

                with torch.no_grad():
                    features, regression, classification, anchors = model(x)

                    regressBoxes = BBoxTransform()
                    clipBoxes = ClipBoxes()

                    out = postprocess(x,
                                    anchors, regression, classification,
                                    regressBoxes, clipBoxes,
                                    threshold, iou_threshold)


                out = invert_affine(framed_metas, out)
                img_id = os.path.basename(image)[:-4]

                # convert result to coco format
                for i in range(len(out)):
                    anns = []
                    _objs = out[i]['class_ids']
                    _bboxes = out[i]['rois']
                    _scores = out[i]['scores']

                    _p = _objs == 0

                    if not len(_p): continue

                    out[i]['class_ids'] = _objs[_p]
                    out[i]['rois'] = _bboxes[_p, :]
                    out[i]['scores'] = _scores[_p]

                    for _d, (bbox, score) in enumerate(zip(_bboxes[_p, :], _scores[_p])):
                        x1, y1, x2, y2 = [int(_) for _ in bbox]
                        if x2-x1 < size_thresh or y2-y1 < size_thresh: continue
                        
                        ann = {
                            'image_id': img_id,
                            'bbox': [x1, y1, x2-x1, y2-y1],
                            'category_id': 1,
                            'track': _d,
                            'name': f'{code:02d}-{_d}',
                            'id': ann_id,
                            'score': float(score)
                        }
                        anns.append(ann)
                all_anns += anns
                
                # display(out, ori_imgs, imshow=True, imwrite=False)
                # if cv2.waitKey(0) == 27: break

            json.dump(all_anns, open(f'det/d{compound_coef}-{code}-iou-{iou_threshold}-thersh-{threshold}.res.json', 'w'))

    # print('running speed test...')
    # with torch.no_grad():
    #     print('test1: model inferring and postprocessing')
    #     print('inferring image for 10 times...')
    #     t1 = time.time()
    #     for _ in range(10):
    #         _, regression, classification, anchors = model(x)

    #         out = postprocess(x,
    #                           anchors, regression, classification,
    #                           regressBoxes, clipBoxes,
    #                           threshold, iou_threshold)
    #         out = invert_affine(framed_metas, out)

    #     t2 = time.time()
    #     tact_time = (t2 - t1) / 10
    #     print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

    #     # uncomment this if you want a extreme fps test
    #     # print('test2: model inferring only')
    #     # print('inferring images for batch_size 32 for 10 times...')
    #     # t1 = time.time()
    #     # x = torch.cat([x] * 32, 0)
    #     # for _ in range(10):
    #     #     _, regression, classification, anchors = model(x)
    #     #
    #     # t2 = time.time()
    #     # tact_time = (t2 - t1) / 10
    #     # print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')
