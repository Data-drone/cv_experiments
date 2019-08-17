# functions and classes to preprocess coco data for object detection

import torch
import torchvision
import os
import random
from torchvision.transforms import functional as F

#   wrapper class for coco detector
#   from https://github.com/pytorch/vision/blob/master/references/detection/coco_utils.py
class CocoDetection(torchvision.datasets.CocoDetection):

    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


# processes coco labels for detection task deployed as transform
# adapted from https://github.com/pytorch/vision/blob/master/references/detection/coco_utils.py
class CocoDetectProcessor(object):

    def __call__(self, image, target):
        
        w, h = image.size
        
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        
        # strip crowd scenes
        anno = [obj for obj in anno if obj['iscrowd'] == 0]
        
        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        
        return image, target


# random flip data processor
class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


# need a custom one to to return the target too
class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


# from https://github.com/pytorch/vision/blob/master/references/detection/transforms.py
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target