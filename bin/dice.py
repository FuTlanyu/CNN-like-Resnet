
'''
获取生成的每一张概率图，及其对应的ground truth（tumor mask），计算Dice系数

python E:\python\program\dachuang_heatmap\bin\dice.py
E:\python\program\dachuang_heatmap\data\prob_map\probmap_l4\test\
F:\LungCancerData\testForHeatmap\test\test_mask\
'''

import sys
import os
import argparse
import glob
import openslide

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Calculate the Dice coefficient of the probability map and the ground truth.')

parser.add_argument('prob_map_path', default=None, metavar='PROB_MAP_PATH', type=str, help='Path to the probability map file')

parser.add_argument('tumor_mask_path', default=None, metavar='TUMOR_MASK_PATH', type=str, help='Path to the tumor mask file')

parser.add_argument('--level', default=4, type=int, help='at which WSI level probability map is generated')


def dice_coeff(pred, gt):
    # smooth = 1.
    # num = pred.size(0)
    # m1 = pred.view(num, -1)  # Flatten
    # m2 = gt.view(num, -1)  # Flatten
    # intersection = (m1 * m2).sum()
    #
    # return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    intersection = (pred * gt).sum()
    return (2. * intersection ) / (pred.sum() + gt.sum())

def normalize(array):
    for x in array:
        x = float(x - np.min(array)) / (np.max(array) - np.min(array))

    return array



def main():
    probmap_paths = glob.glob(os.path.join(args.prob_map_path, '*.npy'))
    probmap_paths.sort()
    tumor_mask_paths = glob.glob(os.path.join(args.tumor_mask_path, '*.tif'))
    tumor_mask_paths.sort()

    prob_gt_pair = zip(probmap_paths, tumor_mask_paths)
    prob_gt_pair = list(prob_gt_pair)

    dice = 0
    count = 0

    for probmap_path, tumor_mask_path in prob_gt_pair:
        prob_map = np.load(probmap_path)

        tumor_mask = openslide.OpenSlide(tumor_mask_path)

        tumor_mask_8 = tumor_mask.read_region((0, 0), args.level, tumor_mask.level_dimensions[args.level])
        tumor_mask_8 = np.transpose(np.array(tumor_mask_8.convert('RGB')), axes=[1, 0, 2])

        tumor_mask_8 = tumor_mask_8[:, :, 1]

        dice += dice_coeff(prob_map, tumor_mask_8)
        count = count + 1

    print('For %d image(s), the dice coefficient is %.4f(%f / %d)\n' %(count, dice/count, dice, count))

if __name__ == '__main__':
    args = parser.parse_args()
    main()