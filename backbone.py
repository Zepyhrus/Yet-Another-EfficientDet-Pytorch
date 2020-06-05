# Author: Zylo117
from abc import ABC, abstractmethod
import math

import torch
from torch import nn

from efficientdet.model import BiFPN, Regressor, Classifier, EfficientNet
from efficientdet.utils import Anchors


class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef])
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef])

        self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef], **kwargs)

        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        max_size = inputs.shape[-1]

        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)

        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(inputs, inputs.dtype)

        return features, regression, classification, anchors

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')



class BaseDetector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def image_preprocess(self, img_name):
        pass

    @abstractmethod
    def image_detection(self, imgs, orig_dim_list):
        pass

    @abstractmethod
    def detect_one_img(self, img_name):
        pass




# we shall create a EffDetector here
class EffDetector(BaseDetector):
  input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

  def __init__(self, compound_coef):
    self.model = None
    self.compound_coef = compound_coef
    self.iou_threshold = 0.4
    self.threshold = 0.2
    self.force_input_size = None
    self.input_size = self.input_sizes[self.compound_coef] if self.force_input_size is None else self.force_input_size
    self.regressBoxes = BBoxTransform()
    self.clipBoxes = ClipBoxes()

  def load_model(self):
    self.model = EfficientDetBackbone(compound_coef=self.compound_coef, num_classes=90)
    self.model.load_state_dict(torch.load(f'weights/efficientdet-d{self.compound_coef}.pth'))
    self.model.requires_grad_(False)
    self.model.eval()
    self.model = self.model.cuda()
    

  def image_preprocess(self, img_source):
    """
    input:
      image name as a string
    return:
      tensor in (1, 3, h, w) mode
    """
    if isinstance(img_source, str):
      img = cv2.imread(img_source)
    else:
      raise IOError('Unknown image source type: {}'.format(type(img_source)))
    
    ori_imgs, framed_imgs, framed_metas = preprocess(img_source, max_size=self.input_size)
    

    return torch.from_numpy(framed_imgs[0]).unsqueeze(0).permute(0, 3, 1, 2)
    


  def image_detection(self, images, im_dim_list):
    """
    input:
      img tensor in (b, 3, h, w) mode, here b refers to batchsize
      im_dim_list in (b, 4):
        [
          [w, h, w, h],
          [w, h, w, h]
        ]
    return:
      box/score tensor in: (batch_index, [x1, y1, x2, y2], score, class_score(1), class_index(0)) mode
    """
    if self.model is None:
      self.load_model()
    
    # we need to recover framed_metas from im_dim_list
    framed_metas = []
    for dim in im_dim_list:
      old_h, old_w = int(dim[1]), int(dim[0])
      if old_w > old_h:
        new_w = self.input_size
        new_h = int(self.input_size / old_w * old_h)
      else:
        new_w = int(self.input_size / old_h * old_w)
        new_h = self.input_size
      
      padding_h = self.input_size - new_h
      padding_w = self.input_size - new_w

      framed_meta = (new_w, new_h, old_w, old_h, padding_w, padding_h)
      framed_metas.append(framed_meta)
    
    with torch.no_grad():
      torch.cuda.synchronize()
      images = images.cuda()

      features, regression, classification, anchors = self.model(images)

      preds = postprocess(
        images, anchors, regression, classification,
        self.regressBoxes, self.clipBoxes,
        self.threshold, self.iou_threshold
      )

      preds = invert_affine(framed_metas, preds)
    
    # interface out, converting EfficientDet to AlphaPose format input
    results = []
    for idx, pred in enumerate(preds):
      for j, bbox in enumerate(pred['rois']):
        if pred['class_ids'][j] != 0: continue
        x1, y1, x2, y2 = tuple(bbox)
        result = [idx, x1, y1, x2, y2, pred['scores'][j], 1, 0]
        results.append(result)

    return torch.tensor(results)

  def detect_one_img(self, img_name):
    """
    note that framed_metas in (new_w, new_h, old_w, old_h, padding_w, padding_h) mode
    """
    ori_imgs, framed_imgs, framed_metas = preprocess(img_name, max_size=self.input_size)

    x = torch.stack([torch.from_numpy(_) for _ in framed_imgs], 0).permute(0, 3, 1, 2)
    x = x.cuda()  # convert to (n, c, h, w)
    
    with torch.no_grad():
      features, regression, classification, anchors = self.model(x)

      preds = postprocess(
        x, anchors, regression, classification,
        self.regressBoxes, self.clipBoxes,
        self.threshold, self.iou_threshold
      )
            
      preds = invert_affine(framed_metas, preds)[0]  # since we only have 1 image here
    
    dets_results = []

    if len(preds['rois']) == 0:
      return dets_results
    
    for i in range(len(preds['rois'])):
      if preds['class_ids'][i] != 0: continue  # we only consider human at this time of view

      # ensamble results
      det_dict = {}
      (x1, y1, x2, y2) = preds['rois'][i]
      x, y, w, h = x1, y1, x2-x1, y2-y1
      det_dict['category_id'] = 1
      det_dict['score'] = preds['scores'][i]
      det_dict['bbox'] = [x, y, w, h]
      det_dict['image_id'] = img_name
      dets_results.append(det_dict)
    
    return dets_results
    