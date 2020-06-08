import os

import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['figure.autolayout'] = True
import cv2
from torchvision import transforms


from efficientdet.dataset import CocoDataset, Resizer

from utils.utils import imread
from train import get_args, Params






if __name__ == "__main__":
  opt = get_args()
  input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

  params = Params(f'projects/{opt.project}.yml')

  training_set = CocoDataset(
    root_dir=os.path.join(opt.data_path, params.project_name),
    set=params.train_set,
    transform=transforms.Compose([
      # Normalizer(mean=params.mean, std=params.std),
      # Augmenter(),
      Resizer(input_sizes[opt.compound_coef])
    ]),
    output_size=input_sizes[opt.compound_coef],
    train=True
  )


  for res in training_set:
    img = res['img']
    bboxes = res['annot'].numpy().astype(int)

    img = (img.numpy() * 255).astype(np.uint8)

    x1s = bboxes[:, 0]
    y1s = bboxes[:, 1]
    x2s = bboxes[:, 2]
    y2s = bboxes[:, 3]

    for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s):
      cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


    cv2.imshow('_', img[:, :, ::-1])
    if cv2.waitKey(0) == 27: break




