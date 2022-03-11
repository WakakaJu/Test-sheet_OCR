import cv2
import numpy as np
from paddleocr import PaddleOCR
from project.Hough_lines import Hough_lines
from project.word_enhance import word_enhance
from project.ocr_paddle import ocr_recognize

class content_pp:
    def ppocr_content(self, image):
        # 实例化
        Hough = Hough_lines()
        enhance = word_enhance()
        ocr = ocr_recognize()

        # 载入模型
        ocr_model = PaddleOCR(use_angle_cls=True, lang="ch")

        block_list = []
        content_th = 0  # 最宽的一个为主要项目内容，找出来
        points_hor, points_ver = Hough.find_lines(image)  # 找出横竖线标点
        for start_hor in range(len(points_hor)):
            if start_hor + 1 >= len(points_hor):
                break
            block = Hough.cut_image(image, points_hor, start_hor, 0)  # block为横条灰度，一条
            if block is None:
                continue
            block_list.append(block)

        for start in range(len(block_list)):
            if start + 1 >= len(block_list):
                break
            if block_list[start + 1].shape[0] >= block_list[content_th].shape[0]:
                content_th = start + 1

        content = block_list[content_th]  # content即为主要项目内容

        content_part = []
        text_part = []
        if len(points_ver) > 0:
            points_ver.insert(0, [0, 0, 0, image.shape[0]])
            for start_ver in range(len(points_ver)):
                if start_ver + 1 >= len(points_ver):
                    break
                if sum(sum(content[:, points_ver[start_ver][0]: points_ver[start_ver + 1][0]])) > 0.25 * (
                        (points_ver[start_ver + 1][0] - points_ver[start_ver + 1][0]) * content.shape[0]):
                    part = Hough.cut_image(content, points_ver, start_ver, 1)
                    if part is not None:
                        content_part.append(part)
                        en_part = enhance.upsampling_piece(part)
                        _, text, _ = ocr.ppocr_recognize(en_part, ocr_model)
                        text_part.append(text)
        return text_part


# imPath = 'E:\文档\处方单识别\demo\screen2.jpg'
# image = cv2.imdecode(np.fromfile(imPath, dtype=np.uint8), -1)
# outPath = 'E:\文档\处方单识别\project\output\\'
# out_name = 'result_xiao'
# out = outPath + out_name + '.txt'