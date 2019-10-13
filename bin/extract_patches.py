import glob
import os

import cv2
import numpy as np

import utils
from bin.wsi_ops import PatchExtractor
from bin.wsi_ops import WSIOps


'''
wsi_ops: is a class : class WSIOps(object) wsi_ops.py
patch_extractor: PatchExtractor(object): wsi_ops.py
patch_index: 700000
'''
# 从肿瘤区域中分割出小块
def extract_positive_patches_from_tumor_wsi(wsi_ops, patch_extractor, patch_index, wsi_path, mask_path, patch_save_dir):
    wsi_paths = glob.glob(os.path.join(wsi_path, '*.tif'))
    wsi_paths.sort()
    mask_paths = glob.glob(os.path.join(mask_path, '*.tif'))
    mask_paths.sort()

    image_mask_pair = zip(wsi_paths, mask_paths)
    image_mask_pair = list(image_mask_pair)
    # image_mask_pair = image_mask_pair[67:68]

    # patch_save_dir = utils.PATCHES_TRAIN_AUG_POS_PATH if augmentation else utils.PATCHES_TRAIN_POS_PATH
    # patch_prefix = utils.PATCH_AUG_TUMOR_PREFIX if augmentation else utils.PATCH_TUMOR_PREFIX

    for image_path, mask_path in image_mask_pair:
        pre_patch_index = patch_index
        print('extract_positive_patches_from_tumor_wsi(): %s' % utils.get_filename_from_path(image_path))
        # 取出level_used这一层的rgb_image和tumor_gt_mask
        wsi_image, rgb_image, _, tumor_gt_mask, level_used = wsi_ops.read_wsi_tumor(image_path, mask_path)
        assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % image_path
        # bounding_boxes为能够包括轮廓的一个最小的矩形
        bounding_boxes = wsi_ops.find_roi_bbox_tumor_gt_mask(np.array(tumor_gt_mask))

        patch_index = patch_extractor.extract_positive_patches_from_tumor_region(wsi_image, np.array(tumor_gt_mask),
                                                                                 level_used, bounding_boxes,
                                                                                 patch_save_dir, patch_index)
        # 为什么要在700000的基础上加上是肿瘤的小块个数再减去700000
        # print('Positive patch count: %d' % (patch_index - utils.PATCH_INDEX_POSITIVE))
        print('%s Positive patch count: %d' % (utils.get_filename_from_path(image_path), (patch_index - pre_patch_index)))
        wsi_image.close()

    return patch_index






# 从组织区域中分割出正常的小块
def extract_negative_patches_from_tumor_wsi(wsi_ops, patch_extractor, patch_index, wsi_path, mask_path, patch_save_dir):
    wsi_paths = glob.glob(os.path.join(wsi_path, '*.tif'))
    wsi_paths.sort()
    mask_paths = glob.glob(os.path.join(mask_path, '*.tif'))
    mask_paths.sort()

    image_mask_pair = zip(wsi_paths, mask_paths)
    image_mask_pair = list(image_mask_pair)
    # image_mask_pair = image_mask_pair[67:68]

    # patch_save_dir = utils.PATCHES_TRAIN_AUG_NEG_PATH if augmentation else utils.PATCHES_TRAIN_NEG_PATH
    # patch_prefix = utils.PATCH_AUG_NORMAL_PREFIX if augmentation else utils.PATCH_NORMAL_PREFIX

    for image_path, mask_path in image_mask_pair:
        print('extract_negative_patches_from_tumor_wsi(): %s' % utils.get_filename_from_path(image_path))

        # tumor_gt_mask 为肿瘤区域对应掩码
        wsi_image, rgb_image, _, tumor_gt_mask, level_used = wsi_ops.read_wsi_tumor(image_path, mask_path)
        assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % image_path

        # 找出肺癌图像某一层的图像的bounding_box，和其对应的组织区域的掩码
        bounding_boxes, _, image_open = wsi_ops.find_roi_bbox(np.array(rgb_image))

        patch_index = patch_extractor.extract_negative_patches_from_tumor_wsi(wsi_image, np.array(tumor_gt_mask),
                                                                              image_open, level_used,
                                                                              bounding_boxes, patch_save_dir, patch_index)

        # print('Negative patches count: %d' % (patch_index - utils.PATCH_INDEX_NEGATIVE))

        wsi_image.close()

    return patch_index



def extract_patches_from_heatmap_false_region_tumor(wsi_ops, patch_extractor, patch_index, augmentation=False):
    tumor_heatmap_prob_paths = glob.glob(os.path.join(utils.HEAT_MAP_DIR, '*umor*prob.png'))
    tumor_heatmap_prob_paths.sort()
    wsi_paths = glob.glob(os.path.join(utils.TUMOR_WSI_PATH, '*.tif'))
    wsi_paths.sort()
    mask_paths = glob.glob(os.path.join(utils.TUMOR_MASK_PATH, '*.tif'))
    mask_paths.sort()
    assert len(tumor_heatmap_prob_paths) == len(wsi_paths), 'Some heatmaps are missing!'

    image_mask_heatmap_tuple = zip(wsi_paths, mask_paths, tumor_heatmap_prob_paths)
    image_mask_heatmap_tuple = list(image_mask_heatmap_tuple)
    # image_mask_heatmap_tuple = image_mask_heatmap_tuple[32:]

    # delete Tumor slides with mirror(duplicate regions) and incomplete annotation: Tumor_018, Tumor_046, Tumor_054
    delete_index = [17, 45, 53]
    for i in range(len(delete_index)):
        print('deleting: %s' % utils.get_filename_from_path(image_mask_heatmap_tuple[delete_index[i] - i][0]))
        del image_mask_heatmap_tuple[delete_index[i] - i]

    patch_save_dir_pos = utils.PATCHES_TRAIN_AUG_EXCLUDE_MIRROR_WSI_POSITIVE_PATH if augmentation else utils.PATCHES_TRAIN_POSITIVE_PATH
    patch_prefix_pos = utils.PATCH_AUG_TUMOR_PREFIX if augmentation else utils.PATCH_TUMOR_PREFIX
    patch_save_dir_neg = utils.PATCHES_TRAIN_AUG_EXCLUDE_MIRROR_WSI_NEGATIVE_PATH if augmentation else utils.PATCHES_TRAIN_NEGATIVE_PATH
    patch_prefix_neg = utils.PATCH_AUG_NORMAL_PREFIX if augmentation else utils.PATCH_NORMAL_PREFIX
    not_0_255_cnt = 0
    for image_path, mask_path, heatmap_prob_path in image_mask_heatmap_tuple:
        print('extract_patches_from_heatmap_false_region_normal(): %s, %s, %s' %
              (utils.get_filename_from_path(image_path), utils.get_filename_from_path(mask_path),
               utils.get_filename_from_path(heatmap_prob_path)))

        wsi_image, rgb_image, wsi_mask, tumor_gt_mask, level_used = wsi_ops.read_wsi_tumor(image_path, mask_path)
        assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % image_path
        # tumor_gt_mask = cv2.cvtColor(tumor_gt_mask, cv2.COLOR_BGR2GRAY)
        # not_0_255_cnt += (tumor_gt_mask[tumor_gt_mask != 255].shape[0]-tumor_gt_mask[tumor_gt_mask == 0].shape[0])
        # print(tumor_gt_mask[tumor_gt_mask != 255].shape[0], tumor_gt_mask[tumor_gt_mask == 0].shape[0], not_0_255_cnt)

        bounding_boxes, image_open = wsi_ops.find_roi_bbox(np.array(rgb_image))

        heatmap_prob = cv2.imread(heatmap_prob_path)
        heatmap_prob = heatmap_prob[:, :, :1]
        heatmap_prob = np.reshape(heatmap_prob, (heatmap_prob.shape[0], heatmap_prob.shape[1]))
        heatmap_prob = np.array(heatmap_prob, dtype=np.float32)
        heatmap_prob /= 255

        patch_index = patch_extractor.extract_patches_from_heatmap_false_region_tumor(wsi_image, wsi_mask,
                                                                                      tumor_gt_mask,
                                                                                      image_open,
                                                                                      heatmap_prob,
                                                                                      level_used, bounding_boxes,
                                                                                      patch_save_dir_pos,
                                                                                      patch_save_dir_neg,
                                                                                      patch_prefix_pos,
                                                                                      patch_prefix_neg,
                                                                                      patch_index)
        print('patch count: %d' % (patch_index - utils.PATCH_INDEX_NEGATIVE))

        wsi_image.close()
        wsi_mask.close()

    # print('not_0_255_cnt: %d' % not_0_255_cnt)
    return patch_index

def extract_patches_from_heatmap_false_region_normal(wsi_ops, patch_extractor, patch_index, augmentation=False):
    normal_heatmap_prob_paths = glob.glob(os.path.join(utils.HEAT_MAP_DIR, 'Normal*prob.png'))
    normal_heatmap_prob_paths.sort()
    wsi_paths = glob.glob(os.path.join(utils.NORMAL_WSI_PATH, '*.tif'))
    wsi_paths.sort()
    assert len(normal_heatmap_prob_paths) == len(wsi_paths), 'Some heatmaps are missing!'

    image_heatmap_tuple = zip(wsi_paths, normal_heatmap_prob_paths)
    image_heatmap_tuple = list(image_heatmap_tuple)
    # image_mask_pair = image_mask_pair[67:68]

    patch_save_dir_neg = utils.PATCHES_TRAIN_AUG_NEGATIVE_PATH if augmentation else utils.PATCHES_TRAIN_NEGATIVE_PATH
    patch_prefix_neg = utils.PATCH_AUG_NORMAL_PREFIX if augmentation else utils.PATCH_NORMAL_PREFIX
    for image_path, heatmap_prob_path in image_heatmap_tuple:
        print('extract_patches_from_heatmap_false_region_normal(): %s, %s' % (utils.get_filename_from_path(image_path)
                                                                              , utils.get_filename_from_path(
            heatmap_prob_path)))

        wsi_image, rgb_image, level_used = wsi_ops.read_wsi_normal(image_path)
        assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % image_path

        bounding_boxes, image_open = wsi_ops.find_roi_bbox(np.array(rgb_image))

        heatmap_prob = cv2.imread(heatmap_prob_path)
        heatmap_prob = heatmap_prob[:, :, :1]
        heatmap_prob = np.reshape(heatmap_prob, (heatmap_prob.shape[0], heatmap_prob.shape[1]))
        heatmap_prob = np.array(heatmap_prob, dtype=np.float32)
        heatmap_prob /= 255

        patch_index = patch_extractor.extract_patches_from_heatmap_false_region_normal(wsi_image,
                                                                                       image_open,
                                                                                       heatmap_prob,
                                                                                       level_used, bounding_boxes,
                                                                                       patch_save_dir_neg,
                                                                                       patch_prefix_neg,
                                                                                       patch_index)
        print('patch count: %d' % (patch_index - utils.PATCH_INDEX_NEGATIVE))

        wsi_image.close()

    return patch_index



def extract_negative_patches_from_normal_wsi(wsi_ops, patch_extractor, patch_index, augmentation=False):
    """
    Extracted up to Normal_060.
    :param wsi_ops:
    :param patch_extractor:
    :param patch_index:
    :param augmentation:
    :return:
    """
    wsi_paths = glob.glob(os.path.join(utils.NORMAL_WSI_PATH, '*.tif'))
    wsi_paths.sort()

    wsi_paths = wsi_paths[61:]

    patch_save_dir = utils.PATCHES_VALIDATION_AUG_NEGATIVE_PATH if augmentation \
        else utils.PATCHES_VALIDATION_NEGATIVE_PATH
    patch_prefix = utils.PATCH_AUG_NORMAL_PREFIX if augmentation else utils.PATCH_NORMAL_PREFIX
    for image_path in wsi_paths:
        print('extract_negative_patches_from_normal_wsi(): %s' % utils.get_filename_from_path(image_path))
        wsi_image, rgb_image, level_used = wsi_ops.read_wsi_normal(image_path)
        assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % image_path

        bounding_boxes, image_open = wsi_ops.find_roi_bbox(np.array(rgb_image))

        patch_index = patch_extractor.extract_negative_patches_from_normal_wsi(wsi_image, image_open,
                                                                               level_used,
                                                                               bounding_boxes,
                                                                               patch_save_dir, patch_prefix,
                                                                               patch_index)
        print('Negative patches count: %d' % (patch_index - utils.PATCH_INDEX_NEGATIVE))

        wsi_image.close()

    return patch_index

if __name__ == '__main__':

    #====================================== train patch extract ===========================================
    # 从肿瘤区域中分割patch
    # num_tumor_patches_train = extract_positive_patches_from_tumor_wsi(WSIOps(), PatchExtractor(), 0,
    #                             utils.TRAIN_WSI_PATH, utils.TRAIN_MASK_PATH, utils.PATCHES_TRAIN_POS_PATH, augmentation=False)
    # print("The number of tumor patch is %d" % num_tumor_patches_train)

    # 从正常组织中分割patch
    # num_normal_patches_train = extract_negative_patches_from_tumor_wsi(WSIOps(), PatchExtractor(), 0,
    #                               utils.TRAIN_WSI_PATH, utils.TRAIN_MASK_PATH, utils.PATCHES_TRAIN_NEG_PATH)
    # print("The number of normal patch is %d" % num_normal_patches_train)


    #===================================== valid patch extract ============================================
    # 从肿瘤区域中分割patch
    num_tumor_patches_valid = extract_positive_patches_from_tumor_wsi(WSIOps(), PatchExtractor(), 0,
                                utils.VALID_WSI_PATH, utils.VALID_MASK_PATH, utils.PATCHES_VALID_POS_PATH)
    print("The number of tumor patch is %d" % num_tumor_patches_valid)

    # 从正常组织中分割patch
    # num_normal_patches_valid = extract_negative_patches_from_tumor_wsi(WSIOps(), PatchExtractor(), 0,
    #                             utils.VALID_WSI_PATH, utils.VALID_MASK_PATH, utils.PATCHES_VALID_NEG_PATH)
    # print("The number of normal patch is %d" % num_normal_patches_valid)






