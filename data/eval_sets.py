
import os
import torch.utils.data as data
from os import listdir
from os.path import join
from data.util import *
import torch.nn.functional as F

class SICEDatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(SICEDatasetFromFolderEval, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        print("------------------------------------------input = ", input)
        _, file = os.path.split(self.data_filenames[index])

        if self.transform:
            input = self.transform(input)
            factor = 8
            h, w = input.shape[1], input.shape[2]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input = F.pad(input.unsqueeze(0), (0,padw,0,padh), 'reflect').squeeze(0)
        return input, file, h, w

    def __len__(self):
        return len(self.data_filenames)

# class SICEDatasetFromFolderEval_dual(data.Dataset):
#     def __init__(self, data_dir, transform=None):
#         super(SICEDatasetFromFolderEval_dual, self).__init__()
#         data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
#         data_filenames.sort()
#         self.data_filenames = data_filenames
#         self.transform = transform

#     def __getitem__(self, index):
#         input = load_img(self.data_filenames[index])
#         print("------------------------------------------input = ", input)
#         _, file = os.path.split(self.data_filenames[index])

#         if self.transform:
#             input = self.transform(input)
#             factor = 8
#             h, w = input.shape[1], input.shape[2]
#             H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
#             padh = H - h if h % factor != 0 else 0
#             padw = W - w if w % factor != 0 else 0
#             input = F.pad(input.unsqueeze(0), (0,padw,0,padh), 'reflect').squeeze(0)
#         return input, file, h, w

#     def __len__(self):
#         return len(self.data_filenames)

import random
import torch
import numpy as np

# TODO edit start    
class SICEDatasetFromFolderEval_dual(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(SICEDatasetFromFolderEval_dual, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        while True:
            fill_index = str(index+1)
            train, tail = os.path.split(self.data_dir)
            folder = join(self.data_dir, fill_index)
            data_gt = join(train+'/label', fill_index+'.JPG')
            if os.path.exists(folder): 
                data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
                num = len(data_filenames)
                break
            else:
                continue

        # 随机挑两张亮度图
        # index1, index2 = random.sample(range(num), 2)

        im1 = load_img(data_filenames[0])
        im2 = load_img(data_filenames[1])
        im_gt = load_img(data_gt)

        # 文件名
        _, file1 = os.path.split(data_filenames[0])
        _, file2 = os.path.split(data_filenames[1])
        _, file_gt = os.path.split(data_gt)

        if self.transform:
            im1 = self.transform(im1)

            im2 = self.transform(im2)

            im_gt = self.transform(im_gt)

        return im1, im2, im_gt, file1, file2, file_gt

    def __len__(self):
        return len([d for d in listdir(self.data_dir) if os.path.isdir(join(self.data_dir,d))])
            
    
class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        if self.transform:
            input = self.transform(input)
        return input, file

    def __len__(self):
        return len(self.data_filenames)