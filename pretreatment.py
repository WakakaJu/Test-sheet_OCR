import cv2
import numpy as np

class pretreat:

    # opencv进行阈值二值化预操作
    def binary(self, image):
        # 输入为彩色图像
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)
            thresh = np.array(thresh)
            return gray, thresh
        # 输入为灰度图(Hough_lines检测输出的是灰度)
        else:
            gray = image
            ret, thresh = cv2.threshold(image, 100, 255, cv2.THRESH_OTSU)
            return gray, thresh