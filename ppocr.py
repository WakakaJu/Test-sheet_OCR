import cv2
from paddleocr import PaddleOCR
from project.cut_lines import cut_lines
from project.Hough_lines import Hough_lines
from project.word_enhance import word_enhance
from project.ocr_paddle import ocr_recognize

class ppocr:
    def ppocr(self, image):
        # 实例化
        Hough = Hough_lines()
        cut = cut_lines()
        enhance = word_enhance()
        ocr = ocr_recognize()

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
        text_part, patch_part = [[] for x in range(2)]

        # 加载模型
        ocr_model = PaddleOCR(use_angle_cls=True, lang="ch")

        for part in content_part:
            text_bar, patch_bar = [[] for x in range(2)]
            _, thresh_part = cv2.threshold(part, 160, 255, cv2.THRESH_OTSU)
            thresh_part = 255 - thresh_part
            bar_list = cut.Cut(thresh_part, 0)
            for bar in bar_list:
                patch_one = []
                text_one = []
                start, end = cut.find_block(bar)
                for start_i, end_i in zip(start, end):
                    patch = bar[:, start_i: end_i]
                    en_patch = enhance.upsampling_piece(patch)
                    patch_one.append(en_patch)
                    _,text, _ = ocr.ppocr_recognize(en_patch, ocr_model)
                    text_one.append(text)
                patch_bar.append(patch_one)
                text_bar.append(text_one)
            patch_part.append(patch_bar)
            text_part.append(text_bar)
        return text_part
