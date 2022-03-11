import cv2
import numpy as np
from project.cut_lines import cut_lines
from project.pretreatment import pretreat as pre

class word_enhance:
    # 边缘填充(填充宽度:默认100)
    def edge_fill(self, img, thickness=100):
        top_size, bottom_size, left_size, right_size = thickness, thickness, thickness, thickness
        filled = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=255)
        return filled

    # opencv上采样(填充宽度默认20)
    def upsampling_piece(self, img,thickness=20):
        enhanced_img = cv2.pyrUp(self.edge_fill(img,thickness))
        return enhanced_img
