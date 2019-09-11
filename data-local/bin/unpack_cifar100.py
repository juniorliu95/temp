# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import re
import os
import pickle
import sys

from tqdm import tqdm
from torchvision.datasets import CIFAR100
import matplotlib.image
import numpy as np


work_dir = os.path.abspath(sys.argv[1])
test_dir = os.path.abspath(os.path.join(sys.argv[2], 'test'))
train_dir = os.path.abspath(os.path.join(sys.argv[2], 'train+val'))

label_1000_dir = os.path.abspath(os.path.join(sys.argv[3], '1000_balanced_labels'))
label_4000_dir = os.path.abspath(os.path.join(sys.argv[3], '4000_balanced_labels'))

cifar100 = CIFAR100(work_dir, download=True)


def load_file(file_name):
    with open(os.path.join(work_dir, cifar100.base_folder, file_name), 'rb') as meta_f:
        return pickle.load(meta_f, encoding="latin1")


def unpack_data_file(source_file_name, target_dir, start_idx):
    print("Unpacking {} to {}".format(source_file_name, target_dir))
    data = load_file(source_file_name)

    for idx, (image_data, label_idx) in tqdm(enumerate(zip(data['data'], data['fine_labels'])), total=len(data['data'])):
        subdir = os.path.join(target_dir, label_names[label_idx])
        name = "{}_{}.png".format(start_idx + idx, label_names[label_idx])
        os.makedirs(subdir, exist_ok=True)
        image = np.moveaxis(image_data.reshape(3, 32, 32), 0, 2)
        matplotlib.image.imsave(os.path.join(subdir, name), image)
    return len(data['data'])

label_names = load_file('meta')['fine_label_names']
print("Found {} label names: {}".format(len(label_names), ", ".join(label_names)))

start_idx = 0
for source_file_path, _ in cifar100.test_list:
    start_idx += unpack_data_file(source_file_path, test_dir, start_idx)

start_idx = 0
for source_file_path, _ in cifar100.train_list:
    start_idx += unpack_data_file(source_file_path, train_dir, start_idx)

print('start preparing labels.....')
import random
for i in range(10):
    ind_1000 = random.sample(range(500), 10)
    ind_4000 = random.sample(range(500), 40)
    list_dir = os.listdir(train_dir) # class folders
    f1000 = open(os.path.join(label_1000_dir,'0'+str(i)+'.txt'), 'w')
    f4000 = open(os.path.join(label_4000_dir,'0'+str(i)+'.txt'), 'w')
    for class_file in list_dir:
        img_files = os.listdir(os.path.join(train_dir, class_file))
        for ind, img in enumerate(img_files):
            if ind in ind_1000:
                f1000.write(os.path.basename(img) + ' ' + class_file + '\n')
            if ind in ind_4000:
                f4000.write(os.path.basename(img) + ' ' + class_file + '\n')
    f1000.close()
    f4000.close()




