'''
将wsi以及对应的tissue_mask在训练好的模型下生成probability map，存储在对应的文件夹下
Parameter:
probs_map.py
F:\LungCancerData\testForHeatmap\test\test_wsi\
E:\python\program\dachuang_heatmap\save\resnet18-save\best.ckpt
E:\python\program\dachuang_heatmap\save\resnet18-save\cnn.json
E:\python\program\dachuang_heatmap\data\test\test_tissue_mask_l4\
E:\python\program\dachuang_heatmap\data\prob_map\probmap_l4\test\
'''

import sys
import os
import argparse
import logging
import json
import time
import glob
import utils

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models
from torch import nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from bin.wsi_producer import WSIPatchDataset  # noqa


parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                 ' patch predictions given a WSI')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('ckpt_path', default=None, metavar='CKPT_PATH', type=str,
                    help='Path to the saved ckpt file of a pytorch model')
parser.add_argument('cnn_path', default=None, metavar='CNN_PATH', type=str,
                    help='Path to the config file in json format related to'
                    ' the ckpt file')
parser.add_argument('mask_path', default=None, metavar='MASK_PATH', type=str,
                    help='Path to the tissue mask of the input WSI file')
parser.add_argument('probs_map_path', default=None, metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
parser.add_argument('--GPU', default='0', type=str, help='which GPU to use'
                    ', default 0')
# --num_workers default = 5
parser.add_argument('--num_workers', default=0, type=int, help='number of '
                    'workers to use to make batch, default 5')
parser.add_argument('--eight_avg', default=0, type=int, help='if using average'
                    ' of the 8 direction predictions for each patch,'
                    ' default 0, which means disabled')


def chose_model(mod):
    if mod == 'resnet18':
        model = models.resnet18(pretrained=False)
    else:
        raise Exception("I have not add any models. ")
    return model


def get_probs_map(model, dataloader):
    probs_map = np.zeros(dataloader.dataset._mask.shape)
    num_batch = len(dataloader)

    count = 0
    time_now = time.time()
    for (data, x_mask, y_mask) in dataloader:
        data = Variable(data.cuda(async=True), volatile=True)
        output = model(data)
        # because of torch.squeeze at the end of forward in resnet.py, if the
        # len of dim_0 (batch_size) of data is 1, then output removes this dim.
        # should be fixed in resnet.py by specifying torch.squeeze(dim=2) later
        if len(output.shape) == 1:
            probs = output.sigmoid().cpu().data.numpy().flatten()
        else:
            probs = output[:,
                           :].sigmoid().cpu().data.numpy().flatten()
        probs_map[x_mask, y_mask] = probs
        count += 1

        time_spent = time.time() - time_now
        time_now = time.time()
        logging.info(
            '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                dataloader.dataset._rotate, count, num_batch, time_spent))

    return probs_map


def make_dataloader(args, cnn, image_path, tissue_mask_path, flip='NONE', rotate='NONE'):
    batch_size = cnn['batch_size'] * 2
    num_workers = args.num_workers

    dataloader = DataLoader(
        WSIPatchDataset(image_path, tissue_mask_path,
                        image_size=cnn['image_size'],
                        crop_size=cnn['crop_size'], normalize=True,
                        flip=flip, rotate=rotate),
        batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)

    with open(args.cnn_path) as f:
        cnn = json.load(f)
    ckpt = torch.load(args.ckpt_path)
    model = chose_model(cnn['model'])
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, 1)
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda().eval()

    wsi_paths = glob.glob(os.path.join(args.wsi_path, '*.tif'))
    wsi_paths.sort()
    tissue_mask_paths = glob.glob(os.path.join(args.mask_path, '*.npy'))
    tissue_mask_paths.sort()

    image_mask_pair = zip(wsi_paths, tissue_mask_paths)
    image_mask_pair = list(image_mask_pair)

    # level 8 ??? tissue_mask
    for image_path, tissue_mask_path in image_mask_pair:
        tissue_mask = np.load(tissue_mask_path)

        if not args.eight_avg:
            dataloader = make_dataloader(args, cnn, image_path, tissue_mask_path, flip='NONE', rotate='NONE')
            # 生成的probability map的位置需要修改
            probs_map = get_probs_map(model, dataloader)
        else:
            probs_map = np.zeros(tissue_mask.shape)

            dataloader = make_dataloader( args, cnn, image_path, tissue_mask_path, flip='NONE', rotate='NONE')

            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(args, cnn, image_path, tissue_mask_path, flip='NONE', rotate='ROTATE_90')

            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(args, cnn, image_path, tissue_mask_path, flip='NONE', rotate='ROTATE_180')

            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(args, cnn, image_path, tissue_mask_path, flip='NONE', rotate='ROTATE_270')

            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(args, cnn, image_path, tissue_mask_path,flip='FLIP_LEFT_RIGHT', rotate='NONE')

            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(args, cnn, image_path, tissue_mask_path, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_90')

            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(args, cnn, image_path, tissue_mask_path, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_180')

            probs_map += get_probs_map(model, dataloader)

            dataloader = make_dataloader(args, cnn, image_path, tissue_mask_path, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_270')

            probs_map += get_probs_map(model, dataloader)

            probs_map /= 8

        # save probability map in the proper path
        np.save(args.probs_map_path + utils.get_filename_from_path(image_path) + '.npy', probs_map)


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
