### Fastai v2 training script
# built on fastai v2.2.2
# testing to see how the presets compare to my hand tuning

## So this is training better than my pytorch lightning...


from fastai.vision.all import *

path = '../cv_data/cifar10'

### Setup Image transforms
item_transforms = [ToTensor, Resize(size=(300,300)), 
              RandomCrop(size=(250,250))
             ]

batch_transforms = [Dihedral(), Normalize()]


### Setup Data Loaders
dls = ImageDataLoaders.from_folder(path, train='train', 
                                   valid='test', device=1, 
                                   item_tfms=item_transforms,
                                  batch_tfms=batch_transforms,
                                  bs=164)

### Setup CNN Learner
learn = cnn_learner(dls, resnet18, pretrained=False, 
                    metrics=[accuracy, top_k_accuracy])

learn.fit(n_epoch=50)