# simple script for removing exif data in training and validation images
# issues that can be caused are: https://github.com/codelucas/newspaper/issues/542

import glob
import piexif

from multiprocessing import Pool
from multiprocessing import cpu_count

dirs = ['../external_data/ImageNet/ILSVRC2012_img_train/**/*.JPEG',
        '../external_data/ImageNet/ILSVRC2012_img_val/**/*.JPEG']

def clean_file(file_name):
    piexif.remove(file_name)


for folder in dirs:
    file_list = glob.glob(folder, recursive=True)

    pool = Pool(cpu_count()-1)
    results = pool.map(clean_file, file_list)
    pool.close()  # 'TERM'
    pool.join() 

    #print(len(file_list))
    
    #print(file_list[:2])



    """
    for filename in glob.iglob(folder, recursive=True):
        nfiles = nfiles + 1
        print("About to process file %d, which is %s." % (nfiles,filename))
        piexif.remove(filename)

    """
    