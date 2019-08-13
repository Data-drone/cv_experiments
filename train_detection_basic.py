# A train script using just pytroch and torchvision
# test to get training loop right

import torch
import torchvision
import torchvision.models.detection as models

### Data Loaders
coco_root = os.path.join('..','..','external_data','coco')
coco_train_set = torchvision.datasets.CocoDetection(root=os.path.join(coco_root, 'train2017'), 
                               annFile=os.path.join(coco_root, 'annotations', 'instances_train2017.json'),
                              transform = torchvision.transforms.ToTensor())

coco_val_set = torchvision.datasets.CocoDetection(root=os.path.join(coco_root, 'val2017'), 
                               annFile=os.path.join(coco_root, 'annotations', 'instances_val2017.json'),
                              transform = torchvision.transforms.ToTensor())


### Need a routine to transform the data?
# batch up the datasets by resolution
# resize routine on this

### pytorch dataloaders
# cannot increase batch size till we sort the resolutions
train_loader = torch.utils.data.DataLoader(coco_train_set,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=4)

val_loader = torch.utils.data.DataLoader(coco_val_set,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=4)

# instantiate model
model = model_test = models.__dict__['fasterrcnn_resnet50_fpn'](pretrained=False)
model.train()

# need declare optimisers

# main train loop
# need to wrap up in a train function


for i, data in enumerate(train_loader):

    image, ann = data

    print(image.shape)

    for item in ann: 
        item["boxes"] = item["bbox"]
        item["labels"] = item["category_id"]
        item["boxes"] = torch.Tensor(item["boxes"]).unsqueeze(dim=0)
        item["labels"] = torch.tensor(item["labels"], dtype=torch.int64)

    losses = model([image], ann)

    # process losses

    # optimizer grad

    # optimizer step


