# A train script using just pytroch and torchvision
# test to get training loop right

import torch
import torchvision
import torchvision.models.detection as models

from data_prep.preproc_coco_detect import CocoDetection, CocoDetectProcessor, coco_remove_images_without_annotations
from data_prep.preproc_coco_detect import Compose, RandomHorizontalFlip, ToTensor
from misc_utils.detection_logger import Logger
import os

import wandb
wandb.init(project="object_detection")

### trigger cuda
device = 'cuda'

### Data Loaders
coco_root = os.path.join('..','external_data','coco')

train_transforms = Compose([CocoDetectProcessor(), ToTensor(), RandomHorizontalFlip(0.5)])
val_transforms = Compose([CocoDetectProcessor(), ToTensor()])

### Coco DataSet Processors
train_set = CocoDetection(os.path.join(coco_root, 'train2017'),
                        os.path.join(coco_root, 'annotations', 'instances_train2017.json'), 
                        train_transforms)
val_set = CocoDetection(os.path.join(coco_root, 'val2017'), 
                    os.path.join(coco_root, 'annotations', 'instances_val2017.json'), 
                    val_transforms)

train_set = coco_remove_images_without_annotations(train_set)

# Coco Dataset Samplers
train_sampler = torch.utils.data.RandomSampler(train_set)
test_sampler = torch.utils.data.SequentialSampler(val_set)
train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, 3, drop_last=True)

# this was in the reference code base but need to unpack why?
# the data has to be converted back to list for the detector later anyway?
def collate_fn(batch):
    return tuple(zip(*batch))

### pytorch dataloaders
# cannot increase batch size till we sort the resolutions
train_loader = torch.utils.data.DataLoader(
        train_set, batch_sampler=train_batch_sampler, num_workers=1,
        collate_fn=collate_fn)

test_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1,
        sampler=test_sampler, num_workers=1,
        collate_fn=collate_fn)


# instantiate model
model = model_test = models.__dict__['fasterrcnn_resnet50_fpn'](pretrained=False)
model.to(device)

## declare optimiser
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
        params, lr=0.0025, momentum=0.9, weight_decay=1e-4)


def batch_loop(model, optimizer, data_loader, device, epoch):
    # based on the train_one_epoch detection engine reference script
    model.train()
    metric_logger = Logger()
    header = 'Epoch: [{}]'.format(epoch)

    for images, targets in metric_logger.log(data_loader, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # converting tensors to numbers
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))

        wandb.log(loss_dict)



def train(model, optimizer, data_loader, test_loader, device):

    for epoch in range(10):

        print(epoch)
        batch_loop(model, optimizer, data_loader, device, epoch)

        # eval loop as well

#wandb.watch(model)
train(model, optimizer, train_loader, test_loader, device)