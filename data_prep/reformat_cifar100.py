### write out cifar100 images
import os
import cv2
import numpy as np
import imageio

# this script unpacks the cifar100 pickles into the format:
## data/cifar-100-oythin/train/<class>/file_x.jpg

data_dir = os.path.join('..','cv_data')
source_data_dir = os.path.join(data_dir, 'cifar-100-python')
train_file = os.path.join(source_data_dir, 'train')
test_file = os.path.join(source_data_dir, 'test')
meta_file = os.path.join(source_data_dir, 'meta')


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def make_dir_if_no_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

def process_train():
    """
    process the cifar 100 train data from the torchvision library 
    into an imagenet format

    /train/<class>/<img>.jpg

    """

    train_entry = unpickle(train_file)
    train_dataset = train_entry[b'data']
    train_targets = train_entry[b'fine_labels'] # will need to edit for coarse
    train_dataset = np.vstack(train_dataset).reshape(-1, 3, 32, 32)
    train_dataset = train_dataset.transpose((0, 2, 3, 1))  

    meta_entry = unpickle(meta_file)
    meta_entry[b'fine_label_names']

    root_path = data_dir + '/cifar100/train/'
    for counter, item in enumerate(train_targets):
        make_dir_if_no_exist(root_path+str(item))
        # write data
        img = train_dataset[counter]
        #bgr_image = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        file_path = root_path+str(item)+'/'+"train_img_{0}.jpg".format(str(counter))
        #print(file_path)
        # something breaks here
        #cv2.imwrite(file_path, bgr_image)
        imageio.imwrite(file_path, img)

def process_test():
    """
    process the cifar 100 test data from the torchvision library 
    into an imagenet format

    /test/<class>/<img>.jpg

    """

    test_entry = unpickle(test_file)
    test_dataset = test_entry[b'data']
    test_targets = test_entry[b'fine_labels']
    test_dataset = np.vstack(test_dataset).reshape(-1, 3, 32, 32)
    test_dataset = test_dataset.transpose((0, 2, 3, 1)) 

    root_path = data_dir + '/cifar100/test/'
    for counter, item in enumerate(test_targets):
        make_dir_if_no_exist(root_path+str(item))
        # write data
        img = test_dataset[counter]
        #bgr_image = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        file_path = root_path+str(item)+'/'+"test_img_{0}.jpg".format(str(counter))
        #print(file_path)
        # something breaks here
        #cv2.imwrite(file_path, bgr_image)
        imageio.imwrite(file_path, img)
    

def process_meta():

    meta_entry = unpickle(meta_file)
    meta_entry[b'fine_label_names']

def main():

    process_train()
    process_test()
    
if __name__ == '__main__':
    main()
