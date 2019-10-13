import cv2
import numpy as np
from openslide import OpenSlide, OpenSlideUnsupportedFormatError

import utils as utils

# 取出肿瘤区域的小块，但取出的个数不定，patch_index为什么初始化为70000
class PatchExtractor(object):
    @staticmethod
    def extract_positive_patches_from_tumor_region(wsi_image, tumor_gt_mask, level_used,
                                                   bounding_boxes, patch_save_dir, patch_index):
        """
            =======================================================================================
            Extract positive patches targeting annotated tumor region

            Save extracted patches to desk as .png image files

            :param wsi_image:
            :param tumor_gt_mask:
            :param level_used:
            :param bounding_boxes: list of bounding boxes corresponds to tumor regions
            :param patch_save_dir: directory to save patches into
            :param patch_prefix: prefix for patch name
            :param patch_index:
            :return:
        """

        mag_factor = pow(2, level_used)
        tumor_gt_mask = cv2.cvtColor(tumor_gt_mask, cv2.COLOR_BGR2GRAY) # 0 and 1
        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        for bounding_box in bounding_boxes:
            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1])
            b_x_end = int(bounding_box[0]) + int(bounding_box[2])
            b_y_end = int(bounding_box[1]) + int(bounding_box[3])

            print(b_x_start, b_x_end)
            print(b_y_start, b_y_end)

            '''
            生成闭区间[low,high]上离散均匀分布的整数值;size 为一个向量
            '''
            # X = np.random.random_integers(b_x_start, high=b_x_end, size=utils.NUM_POSITIVE_PATCHES_FROM_EACH_BBOX) # 500 将low，high之间的值均匀划分
            # Y = np.random.random_integers(b_y_start, high=b_y_end, size=utils.NUM_POSITIVE_PATCHES_FROM_EACH_BBOX)
            #=============================================读取patch=======================================
            # 相当于在bounding box中随机取了250000个点
            # for x, y in zip(X, Y):   # [y, x] ?
            for x in range(b_x_start, b_x_end+1):
                for y in range(b_y_start, b_y_end+1):
                    if int(tumor_gt_mask[y, x]) is utils.PIXEL_WHITE:
                        patch = wsi_image.read_region((x * mag_factor, y * mag_factor), 0,
                                                      (utils.PATCH_SIZE, utils.PATCH_SIZE))  # level_used映射到level 0，从level0中取块
                        print(patch_save_dir + str(patch_index) + '.png')
                        # print(' ', x, y)
                        patch.save(patch_save_dir + str(patch_index) + '.png')
                        patch_index += 1
                        patch.close()
            #=======================================================================================
        return patch_index

    @staticmethod
    def extract_negative_patches_from_normal_wsi(wsi_image, image_open, level_used,
                                                 bounding_boxes, patch_save_dir, patch_prefix,
                                                 patch_index):
        """
            Extract negative patches from Normal WSIs

            Save extracted patches to desk as .png image files

            :param wsi_image:
            :param image_open:
            :param level_used:
            :param bounding_boxes: list of bounding boxes corresponds to detected ROIs
            :param patch_save_dir: directory to save patches into
            :param patch_prefix: prefix for patch name
            :param patch_index:
            :return:

        """

        mag_factor = pow(2, level_used)

        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        for bounding_box in bounding_boxes:
            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1])
            b_x_end = int(bounding_box[0]) + int(bounding_box[2])
            b_y_end = int(bounding_box[1]) + int(bounding_box[3])
            X = np.random.random_integers(b_x_start, high=b_x_end, size=utils.NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)
            Y = np.random.random_integers(b_y_start, high=b_y_end, size=utils.NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)

            for x, y in zip(X, Y):
                if int(image_open[y, x]) is not utils.PIXEL_BLACK:
                    patch = wsi_image.read_region((x * mag_factor, y * mag_factor), 0,
                                                  (utils.PATCH_SIZE, utils.PATCH_SIZE))
                    patch.save(patch_save_dir + patch_prefix + str(patch_index), 'PNG')
                    patch_index += 1
                    patch.close()

        return patch_index



# 分离出正常组织中的patch，tumor_gt_mask为病变区域掩码图像，image_open为组织区域掩码图像
    @staticmethod
    def extract_negative_patches_from_tumor_wsi(wsi_image, tumor_gt_mask, image_open, level_used,
                                                bounding_boxes, patch_save_dir, patch_index):

        """
            From Tumor WSIs extract negative patches from Normal area (reject tumor area)
            Save extracted patches to desk as .png image files

            :param wsi_image:
            :param tumor_gt_mask:
            :param image_open: morphological open image of wsi_image
            :param level_used:
            :param bounding_boxes: list of bounding boxes corresponds to tumor regions
            :param patch_save_dir: directory to save patches into
            :param patch_prefix: prefix for patch name
            :param patch_index:
            :return:

        """

        mag_factor = pow(2, level_used)
        tumor_gt_mask = cv2.cvtColor(tumor_gt_mask, cv2.COLOR_BGR2GRAY)
        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        for bounding_box in bounding_boxes:
            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1])
            b_x_end = int(bounding_box[0]) + int(bounding_box[2])
            b_y_end = int(bounding_box[1]) + int(bounding_box[3])
            # X = np.random.random_integers(b_x_start, high=b_x_end, size=utils.NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX) # 100
            # Y = np.random.random_integers(b_y_start, high=b_y_end, size=utils.NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)

            # for x, y in zip(X, Y):
            for x in range(b_x_start, b_x_end+1):
                for y in range(b_y_start, b_y_end+1):
                    if int(image_open[y, x]) is not utils.PIXEL_BLACK and int(tumor_gt_mask[y, x]) is not utils.PIXEL_WHITE:
                        # mask_gt does not contain tumor area
                        patch = wsi_image.read_region((x * mag_factor, y * mag_factor), 0,
                                                      (utils.PATCH_SIZE, utils.PATCH_SIZE))
                        patch.save(patch_save_dir + str(patch_index) + '.png')
                        print(patch_save_dir + str(patch_index) + '.png')
                        patch_index += 1
                        patch.close()

        return patch_index

    @staticmethod
    def extract_patches_from_heatmap_false_region_tumor(wsi_image, wsi_mask, tumor_gt_mask, image_open,
                                                        heatmap_prob,
                                                        level_used, bounding_boxes,
                                                        patch_save_dir_pos, patch_save_dir_neg,
                                                        patch_prefix_pos, patch_prefix_neg,
                                                        patch_index):
        """

            From Tumor WSIs extract negative patches from Normal area (reject tumor area)
                        Save extracted patches to desk as .png image files

            :param wsi_image:
            :param wsi_mask:
            :param tumor_gt_mask:
            :param image_open: morphological open image of wsi_image
            :param heatmap_prob:
            :param level_used:
            :param bounding_boxes: list of bounding boxes corresponds to tumor regions
            :param patch_save_dir_pos: directory to save positive patches into
            :param patch_save_dir_neg: directory to save negative patches into
            :param patch_prefix_pos: prefix for positive patch name
            :param patch_prefix_neg: prefix for negative patch name
            :param patch_index:
            :return:
        """

        mag_factor = pow(2, level_used)
        tumor_gt_mask = cv2.cvtColor(tumor_gt_mask, cv2.COLOR_BGR2GRAY)
        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        for bounding_box in bounding_boxes:
            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1])
            b_x_end = int(bounding_box[0]) + int(bounding_box[2])
            b_y_end = int(bounding_box[1]) + int(bounding_box[3])
            col_cords = np.arange(b_x_start, b_x_end)
            row_cords = np.arange(b_y_start, b_y_end)

            for row in row_cords:
                for col in col_cords:
                    if int(image_open[row, col]) is not utils.PIXEL_BLACK:  # consider pixels from ROI only
                        # extract patch corresponds to false positives
                        if heatmap_prob[row, col] >= utils.TUMOR_PROB_THRESHOLD:
                            if int(tumor_gt_mask[row, col]) == utils.PIXEL_BLACK:
                                # mask_gt does not contain tumor area
                                mask = wsi_mask.read_region((col * mag_factor, row * mag_factor), 0,
                                                            (utils.PATCH_SIZE, utils.PATCH_SIZE))
                                mask_gt = cv2.cvtColor(np.array(mask), cv2.COLOR_BGR2GRAY)
                                white_pixel_cnt_gt = cv2.countNonZero(mask_gt)
                                if white_pixel_cnt_gt == 0:
                                    patch = wsi_image.read_region((col * mag_factor, row * mag_factor), 0,
                                                                  (utils.PATCH_SIZE, utils.PATCH_SIZE))
                                    patch.save(patch_save_dir_neg + patch_prefix_neg + str(patch_index), 'PNG')
                                    patch_index += 1
                                    patch.close()

                                mask.close()
                        # extract patch corresponds to false negatives
                        elif int(tumor_gt_mask[row, col]) is not utils.PIXEL_BLACK \
                                and heatmap_prob[row, col] < utils.TUMOR_PROB_THRESHOLD:
                            # mask_gt does not contain tumor area
                            mask = wsi_mask.read_region((col * mag_factor, row * mag_factor), 0,
                                                        (utils.PATCH_SIZE, utils.PATCH_SIZE))
                            mask_gt = cv2.cvtColor(np.array(mask), cv2.COLOR_BGR2GRAY)
                            white_pixel_cnt_gt = cv2.countNonZero(mask_gt)
                            if white_pixel_cnt_gt >= ((utils.PATCH_SIZE * utils.PATCH_SIZE) * 0.85):
                                patch = wsi_image.read_region((col * mag_factor, row * mag_factor), 0,
                                                              (utils.PATCH_SIZE, utils.PATCH_SIZE))
                                patch.save(patch_save_dir_pos + patch_prefix_pos + str(patch_index), 'PNG')
                                patch_index += 1
                                patch.close()

                            mask.close()

        return patch_index

    @staticmethod
    def extract_patches_from_heatmap_false_region_normal(wsi_image, image_open,
                                                         heatmap_prob,
                                                         level_used, bounding_boxes,
                                                         patch_save_dir_neg,
                                                         patch_prefix_neg,
                                                         patch_index):
        """

            From Tumor WSIs extract negative patches from Normal area (reject tumor area)
                        Save extracted patches to desk as .png image files

            :param wsi_image:
            :param image_open: morphological open image of wsi_image
            :param heatmap_prob:
            :param level_used:
            :param bounding_boxes: list of bounding boxes corresponds to tumor regions
            :param patch_save_dir_neg: directory to save negative patches into
            :param patch_prefix_neg: prefix for negative patch name
            :param patch_index:
            :return:
        """

        mag_factor = pow(2, level_used)
        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        for bounding_box in bounding_boxes:
            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1])
            b_x_end = int(bounding_box[0]) + int(bounding_box[2])
            b_y_end = int(bounding_box[1]) + int(bounding_box[3])
            col_cords = np.arange(b_x_start, b_x_end)
            row_cords = np.arange(b_y_start, b_y_end)

            for row in row_cords:
                for col in col_cords:
                    if int(image_open[row, col]) is not utils.PIXEL_BLACK:  # consider pixels from ROI only
                        # extract patch corresponds to false positives
                        if heatmap_prob[row, col] >= utils.TUMOR_PROB_THRESHOLD:
                            # mask_gt does not contain tumor area
                            patch = wsi_image.read_region((col * mag_factor, row * mag_factor), 0,
                                                          (utils.PATCH_SIZE, utils.PATCH_SIZE))
                            patch.save(patch_save_dir_neg + patch_prefix_neg + str(patch_index), 'PNG')
                            patch_index += 1
                            patch.close()

        return patch_index


class WSIOps(object):
    """
        # ================================
        # Class to annotate WSIs with ROIs
        # ================================

    """

    def_level = 7

    @staticmethod
    def read_wsi_mask(mask_path, level=def_level):
        try:
            wsi_mask = OpenSlide(mask_path)

            mask_image = np.array(wsi_mask.read_region((0, 0), level,
                                                       wsi_mask.level_dimensions[level]))

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return None, None

        return wsi_mask, mask_image

    @staticmethod
    def read_wsi_normal(wsi_path):
        """
            # =====================================================================================
            # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            wsi_image = OpenSlide(wsi_path)
            level_used = wsi_image.level_count - 1
            rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used]))

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return None, None, None

        return wsi_image, rgb_image, level_used

    @staticmethod
    def read_wsi_tumor(wsi_path, mask_path):
        """
            # =====================================================================================
            # read WSI image and resize
            # Due to memory constraint, we use down sampled (8th level, 1/256 resolution) image
            # ======================================================================================
        """
        try:
            wsi_image = OpenSlide(wsi_path)
            wsi_mask = OpenSlide(mask_path)

            level_used = wsi_image.level_count - 1 # 8
            '''
            # read_region(location, level, size)，Return an RGBA Image containing the contents of the specified region.
            # Parameters:  
            # location (tuple) tuple giving the top left pixel in the level 0 reference frame
            # level (int)  the level number
            # size (tuple) (width, height) tuple giving the region size
            # read_region()返回一个数组的rgba数值数组
            '''
            rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used]))

            mask_level = wsi_mask.level_count - 1
            tumor_gt_mask = wsi_mask.read_region((0, 0), mask_level,
                                                 wsi_image.level_dimensions[mask_level])
            # 不需要resize
            resize_factor = float(1.0 / pow(2, level_used - mask_level))
            # print('resize_factor: %f' % resize_factor)
            tumor_gt_mask = cv2.resize(np.array(tumor_gt_mask), (0, 0), fx=resize_factor, fy=resize_factor)

            wsi_mask.close()
        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return None, None, None, None

        return wsi_image, rgb_image, wsi_mask, tumor_gt_mask, level_used

    def find_roi_bbox_tumor_gt_mask(self, mask_image):
        mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY) # 转换为灰度图
        bounding_boxes, _ = self.get_bbox(np.array(mask)) # 得到边缘的边界框
        return bounding_boxes



# 得到组织区域的掩码，对掩码图像先做闭运算后做开运算，得到组织区域的bounding_box
    def find_roi_bbox(self, rgb_image):
        # hsv -> 3 channel
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV) # HSV：色度、饱和度、亮度
        lower_red = np.array([20, 20, 20])
        upper_red = np.array([200, 200, 200])
        # mask -> 1 channel
        # Checks if array elements lie between the elements of two other arrays. 如果三通道的值都在该范围之间，则将其置为255，否则为0
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # 闭运算
        close_kernel = np.ones((20, 20), dtype=np.uint8)
        image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)

        # 开运算
        # 先腐蚀后膨胀
        # 作用：放大裂缝和低密度区域，消除小物体，在平滑较大物体的边界时，不改变其面积
        open_kernel = np.ones((5, 5), dtype=np.uint8)
        image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

        bounding_boxes, rgb_contour = self.get_bbox(image_open, rgb_image=rgb_image) # rgb_contour为加上了边框后的wsi图像
        return bounding_boxes, rgb_contour, image_open

    @staticmethod
    def get_image_open(wsi_path):
        try:
            wsi_image = OpenSlide(wsi_path)
            level_used = wsi_image.level_count - 1
            rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used]))
            wsi_image.close()
        except OpenSlideUnsupportedFormatError:
            raise ValueError('Exception: OpenSlideUnsupportedFormatError for %s' % wsi_path)

        # hsv -> 3 channel
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([20, 20, 20])
        upper_red = np.array([200, 200, 200])
        # mask -> 1 channel
        mask = cv2.inRange(hsv, lower_red, upper_red)

        close_kernel = np.ones((20, 20), dtype=np.uint8)
        image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
        open_kernel = np.ones((5, 5), dtype=np.uint8)
        image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

        return image_open







    @staticmethod
    def get_bbox(cont_img, rgb_image=None):
        '''
        findContours检测图像轮廓
        cv2.RETR_EXTERNAL：只检查外部轮廓
        CV_CHAIN_APPROX_NONE ：保存物体边界上所有连续的轮廓点到contours向量内
        '''
        _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rgb_contour = None
        if rgb_image:
            rgb_contour = rgb_image.copy()
            line_color = (255, 0, 0)  # blue color code
            cv2.drawContours(rgb_contour, contours, -1, line_color, 2)
        # 找到能够包括轮廓的一个最小的矩形
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        return bounding_boxes, rgb_contour

    @staticmethod
    def draw_bbox(image, bounding_boxes):
        rgb_bbox = image.copy()
        for i, bounding_box in enumerate(bounding_boxes):
            x = int(bounding_box[0])
            y = int(bounding_box[1])
            cv2.rectangle(rgb_bbox, (x, y), (x + bounding_box[2], y + bounding_box[3]), color=(0, 0, 255),
                          thickness=2)
        return rgb_bbox

    @staticmethod
    def split_bbox(image, bounding_boxes, image_open):
        rgb_bbox_split = image.copy()
        for bounding_box in bounding_boxes:
            for x in range(bounding_box[0], bounding_box[0] + bounding_box[2]):
                for y in range(bounding_box[1], bounding_box[1] + bounding_box[3]):
                    if int(image_open[y, x]) == 1:
                        cv2.rectangle(rgb_bbox_split, (x, y), (x, y),
                                      color=(255, 0, 0), thickness=2)

        return rgb_bbox_split
