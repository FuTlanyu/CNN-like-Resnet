from PIL import Image
import openslide
import utils
import glob
import os
import numpy as np
import openslide
# image = Image.open('E:/python/program/dachuang_heatmap/data/train/train_patches/0/0.png')
# level 0-8
# ope = openslide.OpenSlide('F:/LungCancerData/testForHeatmap/test/test_wsi/81.tif')
# www = 'fdsg/' + '32' + '.npy'
# print(www)

# 提取出路径中的序号
# wsi_paths = glob.glob(os.path.join('F:LungCancerData\\testForHeatmap\\train\\train_wsi\\', '*.tif'))
# wsi_paths.sort()
# # wsi_paths = list(wsi_paths)
# for wsi_path in wsi_paths:
#     name = utils.get_filename_from_path(wsi_path)
#     print(name)

# 查看概率图属性
# prob = np.load('E:\\python\\program\\dachuang_heatmap\\data\\prob_map\\train\\76.npy')
# # tumor mask
# tumor_mask = openslide.OpenSlide('F:\\LungCancerData\\testForHeatmap\\train\\train_mask\\76.tif')
# tumor_mask_8 = np.transpose(np.array(tumor_mask.read_region( (0, 0), 8 ,tumor_mask.level_dimensions[8]).convert('RGB')), axes=[1, 0, 2])
# tumor_mask_8 = tumor_mask_8[:,:,1]



tissue_mask_l4_81 = np.load('E:\\python\\program\\dachuang_heatmap\\data\\test\\test_tissue_mask_l8\\81.npy')








print('Bravo!')