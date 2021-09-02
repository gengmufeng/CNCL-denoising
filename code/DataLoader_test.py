# -*- coding: utf-8 -*-
"""
Created on 2021-06-14

@author: Mufeng Geng
"""
import os
import torch.utils.data as data
from PIL import Image
import numpy as np

# Path of the data set
path_noisy = r"../data/noisy/"
path_clean = r"../data/clean/"

# Divide the data set according to the txt files
noisy_txt = r"../data/dataset_division/test_noisy.txt"
clean_txt = r"../data/dataset_division/test_clean.txt"

noisy_list = list()
clean_list = list()
for line_noisy in open(noisy_txt, "r"):
    line_noisy = line_noisy[:-1]
    path_noisy_image = os.path.join(path_noisy, line_noisy)
    noisy_list.append(path_noisy_image)

for line_clean in open(clean_txt, "r"):
    line_clean = line_clean[:-1]
    path_clean_image = os.path.join(path_clean, line_clean)
    clean_list.append(path_clean_image)


def get_Test_Set():
    return DatasetFromFolder(noisy_list, clean_list)


def load_image(filepath):
    image = Image.open(filepath)
    image = np.array(image).astype('float32')
    mean = np.mean(image)
    var = np.var(image)
    return np.expand_dims(image, axis=0), mean, var


class DatasetFromFolder(data.Dataset):
    def __init__(self, noisy_list, clean_list):
        super(DatasetFromFolder, self).__init__()
        self.noisy_list = noisy_list
        self.clean_list = clean_list

    def __getitem__(self, index):
        noisy, mean_n, var_n = load_image(self.noisy_list[index])
        clean, mean_c, var_c = load_image(self.clean_list[index])
        noisy = (noisy - mean_n) / var_n
        clean = (clean - mean_n) / var_n
        noise = noisy - clean
        target = np.concatenate((clean, noise), axis=0)
        return {"A": target, "B": noisy, "C": mean_n, "D": var_n}

    def __len__(self):
        return len(self.noisy_list)