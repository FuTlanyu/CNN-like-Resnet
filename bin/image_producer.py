import os
import sys

import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image

np.random.seed(0)

from torchvision import transforms  # noqa

'''
dataset_train = ImageDataset(cnn['data_path_train'],
                                 cnn['image_size'](256),
                                 cnn['crop_size'](224),
                                 cnn['normalize'](True))
dataset_valid = ImageDataset(cnn['data_path_valid'],
                                 cnn['image_size'],
                                 cnn['crop_size'],
                                 cnn['normalize'])
'''
class ImageDataset(Dataset):

    def __init__(self, data_path, img_size,
                 crop_size=224, normalize=True):
        self._data_path = data_path
        self._img_size = img_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        self._pre_process()

    def _pre_process(self):
        # find classes
        if sys.version_info >= (3, 5):
            # Faster and available in python 3.5 and above
            # 获得data_path下各个文件夹的名字
            classes = [d.name for d in os.scandir(self._data_path) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self._data_path) if os.path.isdir(os.path.join(self._data_path, d))]
        classes.sort()
        # class_to_idx 为文件夹名及其对应的索引构成的字典
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        # make dataset
        # items中的每一项为patch路径和其所属的文件夹对应的索引值构成的元组
        self._items = []
        for target in sorted(class_to_idx.keys()):
            # d为每一个文件夹路径
            d = os.path.join(self._data_path, target)
            if not os.path.isdir(d):
                continue

            # fnames 为d路径的下一级目录中所有的文件名构成的列表
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if fname.split('.')[-1] == 'png':
                        # path为每一个patch对应的路径
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        self._items.append(item)

        random.shuffle(self._items)

        self._num_images = len(self._items)

    '''
    items为map-style datasets 需实现__len__()与__getitem__()方法
    详情参见https://pytorch.org/docs/stable/data.html
    '''
    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        path, label = self._items[idx]
        label = np.array(label, dtype=float)

        img = Image.open(path)
        img = img.convert('RGB')
        # color jitter
        img = self._color_jitter(img)

        # use left_right flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # use rotate
        num_rotate = np.random.randint(0, 4)
        img = img.rotate(90 * num_rotate)

        # PIL image: H W C
        # torch image: C H W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0) / 128.0

        return img, label

