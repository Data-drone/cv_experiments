# simple script for removing exif data in training and validation images
# issues that can be caused are: https://github.com/codelucas/newspaper/issues/542

import glob
import piexif

nfiles = 0
dirs = ['../external_data/ImageNet/ILSVRC2012_img_train/**/*.JPEG',
        '../external_data/ImageNet/ILSVRC2012_img_val/**/*.JPEG']

for folder in dirs:
    for filename in glob.iglob(folder, recursive=True):
        nfiles = nfiles + 1
        print("About to process file %d, which is %s." % (nfiles,filename))
        piexif.remove(filename)
