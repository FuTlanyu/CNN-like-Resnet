'''
1.
python E:\python\program\dachuang_heatmap\bin\test_tissue_mask.py
F:/LungCancerData/testForHeatmap/test/test_wsi/
E:/python/program/dachuang_heatmap/data/test/test_tissue_mask_l4/
4
获得wsi路径下的wsi图像在level_4下的组织掩码Tumor_tissue.npy存放在对应的路径中；
'''

import sys
import os
import argparse
import logging
import glob
import utils

import numpy as np
import openslide
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save'
                                 ' it in npy format')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the WSI file')
parser.add_argument('npy_path', default=None, metavar='NPY_PATH', type=str,
                    help='Path to the output npy mask file')
parser.add_argument('--level', default=4, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')
parser.add_argument('--RGB_min', default=50, type=int, help='min value for RGB'
                    ' channel, default 50')


def run(args):
    logging.basicConfig(level=logging.INFO)
    wsi_paths = glob.glob(os.path.join(args.wsi_path, '*.tif'))
    wsi_paths.sort()

    for wsi_path in wsi_paths:
        filename = utils.get_filename_from_path(wsi_path)
        print(filename)

        slide = openslide.OpenSlide(wsi_path)

        # note the shape of img_RGB is the transpose of slide.level_dimensions
        img_RGB = np.transpose(np.array(slide.read_region((0, 0),
                               args.level,
                               slide.level_dimensions[args.level]).convert('RGB')),
                               axes=[1, 0, 2])

        img_HSV = rgb2hsv(img_RGB)

        background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
        background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
        background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
        min_R = img_RGB[:, :, 0] > args.RGB_min
        min_G = img_RGB[:, :, 1] > args.RGB_min
        min_B = img_RGB[:, :, 2] > args.RGB_min

        tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B

        np.save(args.npy_path + filename + '.npy', tissue_mask)


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
