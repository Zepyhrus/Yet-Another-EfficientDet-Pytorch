from glob import glob


import cv2
import numpy as np


import torch
from torch.backends import cudnn


from backbone import EffDetector
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, display




def demo_more(detector, img_sources):
  imgs = torch.cat([detector.image_preprocess(_) for _ in img_sources])
  orig_imgs = [cv2.imread(_) for _ in img_sources]
  im_dim_list = torch.FloatTensor([(_.shape[1], _.shape[0]) for _ in orig_imgs]).repeat(1, 2)


  results = detector.image_detection(imgs, im_dim_list)

  # visualization
  results = results.numpy()
  for idx, img in enumerate(orig_imgs):
    bboxes = results[results[:, 0].astype(np.int) == idx, 1:5]
    scores = results[results[:, 0].astype(np.int) == idx, 5]

    for j, box in enumerate(bboxes):
      x1, y1, x2, y2 = box.astype(np.int)
      score_txt = '{:.3f}'.format(scores[j])

      cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cv2.putText(img, score_txt, (x1, y1+10), 0, 0.75, (0, 255, 0), 2)

    cv2.imshow('_', img)
    if cv2.waitKey(0) == 27: break


def demo_one(detector, img_sources):
  for image in img_sources:
    # if 'web' not in image:
    #   continue
    results = detector.detect_one_img(image)

    img = cv2.imread(image)

    for res in results:
      x, y, w, h = (int(_) for _ in res['bbox'])  # in x,y,w,h mode
      cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
      cv2.putText(img, '{:.3f}'.format(res['score']), (x, y+18), 0, 0.75, (0, 255, 0), 2)

    if img.shape[0] * img.shape[1] >= 1280 * 960:
      ratio = np.sqrt(1280 * 960 / img.shape[0] / img.shape[1])
      img = cv2.resize(img, (int(img.shape[1]*ratio), int(img.shape[0]*ratio)))

    cv2.imshow('_', img)
    if cv2.waitKey(0) == 27: break



if __name__ == "__main__":
  # img_sources = [
  #   './data/seedland/pose_seg_hard/1.jpg',
  #   './data/seedland/pose_seg_hard/2.jpg',
  #   './data/seedland/pose_seg_hard/3.png'
  # ]
  img_sources = glob('./data/seedland/dense/*')

  detector = EffDetector(7)
  detector.load_model()

  # demo_more(detector, img_sources)

  
  # demo on 'detect_one_img'
  demo_one(detector, img_sources)
  
  
  

