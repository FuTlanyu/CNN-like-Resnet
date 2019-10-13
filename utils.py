# Tumor (1) Normal (0)


TRAIN_DATA_DIR = 'E:/python/program/dachuang_heatmap/data/train/'

# TRAIN_MASK_PATH = TRAIN_DATA_DIR + 'train_tissue_mask'
# TRAIN_WSI_PATH = TRAIN_DATA_DIR + 'train_wsi'
TRAIN_MASK_PATH = 'F:/LungCancerData/testForHeatmap/train/train_tissue_mask'
TRAIN_WSI_PATH = 'F:/LungCancerData/testForHeatmap/train/train_wsi'

PATCHES_TRAIN_DIR = TRAIN_DATA_DIR + 'train_patches/'
PATCHES_TRAIN_POS_PATH = PATCHES_TRAIN_DIR + '1/'
PATCHES_TRAIN_NEG_PATH = PATCHES_TRAIN_DIR + '0/'

PATCHES_TRAIN_AUG_DIR = TRAIN_DATA_DIR + 'train_patches_aug/'
PATCHES_TRAIN_AUG_POS_PATH = PATCHES_TRAIN_AUG_DIR + '1/'
PATCHES_TRAIN_AUG_NEG_PATH = PATCHES_TRAIN_AUG_DIR + '0/'


VALID_DATA_DIR = 'E:/python/program/dachuang_heatmap/data/valid/'

VALID_WSI_PATH = 'F:/LungCancerData/testForHeatmap/valid/valid_wsi'
VALID_MASK_PATH = 'F:/LungCancerData/testForHeatmap/valid/valid_mask'

PATCHES_VALID_DIR = VALID_DATA_DIR + 'valid_patches/'
PATCHES_VALID_POS_PATH = PATCHES_VALID_DIR + '1/'
PATCHES_VALID_NEG_PATH = PATCHES_VALID_DIR + '0/'




NUM_POSITIVE_PATCHES_FROM_EACH_BBOX = 500

PIXEL_WHITE = 1
PIXEL_BLACK = 0

PATCH_INDEX_NEGATIVE = 700000
PATCH_INDEX_POSITIVE = 700000

PATCH_SIZE = 256









def get_filename_from_path(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[path_tokens.__len__() - 1].split('.')[0]
    filename = filename.split('\\')[-1]
    return filename








