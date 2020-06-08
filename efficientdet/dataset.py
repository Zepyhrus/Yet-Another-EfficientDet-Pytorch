import os
import random

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2


from utils.utils import imread

from alphapose.utils.noise import random_blur, random_brightness, random_noise, random_pass, random_hsv

class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None, train=True, output_size=512):
        self._output_size = output_size
        self._train = train
        self.cat_ids = [1] # we only condsider human

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform


        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))

        all_image_ids = self.coco.getImgIds()
        human_image_ids = self.coco.getImgIds(catIds=1)
        # empty_image_ids = all_image_ids
        # for h in human_image_ids:
        #     empty_image_ids.remove(h)
        
        # TODO: we use human with images and 10% images without humans
        self.image_ids = human_image_ids # + empty_image_ids[::10]

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)  # here img is still a numpy
        annot = self.load_annotations(idx)
        
        if self._train:
            # add random augumentation
            if np.random.randint(2):
                img = random_brightness(img)
            
            if np.random.randint(2):
                img = random_blur(img)

            if np.random.randint(2):
                img = random_noise(img)

            img = random_hsv(img)

            # add size augumentation
            H, W, C = img.shape
            _drift = self._output_size / 6
            _half = self._output_size / 2
            _amplf = 0.8 + np.random.rand()
            _crop  = 0.5 + np.random.rand()

            # print(_amplf, _crop)

            _drift_x = np.clip(np.random.randn() * _drift, -3*_drift, 3*_drift)
            _drift_y = np.clip(np.random.randn() * _drift, -3*_drift, 3*_drift)

            # print(_drift_x, _drift_y)

            src = np.array([
                [0, 0],
                [0, H],
                [W, 0]
            ], dtype=np.float32)

            dst = np.array([
                [_half + _drift_x - _half*_amplf, _half + _drift_y - _half*H/W*_amplf*_crop],
                [_half + _drift_x - _half*_amplf, _half + _drift_y + _half*H/W*_amplf*_crop],
                [_half + _drift_x + _half*_amplf, _half + _drift_y - _half*H/W*_amplf*_crop]
            ], dtype=np.float32)

            _M = cv2.getAffineTransform(src, dst)

            img = cv2.warpAffine(img, _M, (self._output_size, self._output_size), flags=cv2.INTER_LINEAR)

            # rotate bbox
            annot[:, :2] = np.concatenate((annot[:, :2], np.ones((annot.shape[0], 1))), axis=1) @ _M.T
            annot[:, 2:4] = np.concatenate((annot[:, 2:4], np.ones((annot.shape[0], 1))), axis=1) @ _M.T

            annot = np.clip(annot, 0, self._output_size)

            pick = np.logical_and(
                (annot[:, 2] - annot[:, 0]) >= self._output_size/64,
                (annot[:, 3] - annot[:, 1]) >= self._output_size/64
            )
            
            annot = np.zeros((0, 5)) if pick.sum() == 0 else annot[pick, :]
            # print(annot)


        sample = {'img': img.astype(np.float32) / 255, 'annot': annot}

        if self.transform:
            sample = self.transform(sample)
    
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)[:, :, ::-1]

        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # parse annotations, we only pick annotations contains human and cars
        coco_annotations = self.coco.loadAnns(annotations_ids)
        coco_annotations = [_ for _ in coco_annotations if _['category_id'] in self.cat_ids]

        # some images appear to miss annotations
        if len(coco_annotations) == 0:
            return annotations

        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2 - 1
            annots[:, 2] = cols - x_tmp - 1

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
