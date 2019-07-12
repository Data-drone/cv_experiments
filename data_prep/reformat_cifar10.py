
# quick script to reformat the cifar 10 data
# data courtesy of https://pjreddie.com/projects/cifar-10-dataset-mirror/
# we need to transform that data from:
### /data/train/1_<class>.png
# to
### /data/train/<class>/1_<class>.png

import os
import glob

data_dir = os.path.join('..','data')
source_data_dir = os.path.join(data_dir, 'cifar10')
train_files = os.path.join(source_data_dir, 'train')
test_files = os.path.join(source_data_dir, 'test')
labels = os.path.join(source_data_dir, 'labels.txt')

# create the train/test folders
category_list = [line. rstrip('\n') for line in open(labels)]


for item in category_list:
    create_path_train = os.path.join(train_files, item)
    create_path_test = os.path.join(test_files, item)

    try:
        os.mkdir(create_path_train)
    except FileExistsError:
        print('{0} already exists'.format(create_path_train))

    try:
        os.mkdir(create_path_test)
    except FileExistsError:
        print('{0} already exists'.format(create_path_test))

# get the list of files then move them
train_images = glob.glob(os.path.join(train_files, '*.png'))
test_images = glob.glob(os.path.join(test_files, '*.png'))

# loop through files and move them one by one
def move_file(image_list: list, categories: list) -> None:
    for image_file in image_list:
        image_file_name = image_file.split('/')[-1]
        category = image_file_name.split('_')[-1].split('.')[0]
        path_list = image_file.split('/')
        path_list.pop()
        new_path = os.path.join(*path_list, category, image_file_name)

        assert category in categories

        #print("old: {0} new: {1}".format(image_file, new_path))
        os.rename(image_file,new_path)

# move the files
move_file(train_images, category_list)
move_file(test_images, category_list)
    
