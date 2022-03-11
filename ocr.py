import cv2
import pytesseract
import numpy as np
from project.pretreatment import pretreat
from project.Hough_lines import Hough_lines
from project.cut_lines import cut_lines
from project.word_enhance import word_enhance

class ocr_recognize:

    # 简单识别
    def ocr_tesseract(self, image, psm):
        # 使用谷歌tesseract识别引擎对文字进行识别
        # 参数如下：
        # -l 后接字库，chi_sim识别中文
        # --oem 使用LSTM作为OCR引擎，可选值为0,1,2,3：
        #  0    Legacy engine only
        #  1    Negural nets LSTM engine only
        #  2    Legacy + LSTM engines (不可用)
        #  3    Default,based on what is available.
        #  --psm  设置Page Segmentation模式为自启动
        config = '-l chi_sim --oem 1 --psm ' + '%d' % psm
        # config = '-l normal --oem 1 --psm ' + '%d' % psm

        # 进行识别，本质上式调用tesseract命令行工具
        text = pytesseract.image_to_string(image, config = config)
        return text

    # 对整个图像切割(切割到每个字)，对字识别，再整合
    def merge_recognize(self, image, out):
        # pre = pretreat()
        cut = cut_lines()
        enhance = word_enhance()
        Hough = Hough_lines()
        # img_Hough = Hough.cut_image(img)
        # bar_list = cut.Cut(image, 0)  # 横切条形，所有条列表
        text_list = []
        text_piece_list = []
        piece_list_total = []
        enhanced_list = []
        enhanced_piece_list = []
        points_hor = Hough.find_lines(image)
        k = 0
        for start in range(len(points_hor)):
            kk = 0
            if start + 1 >= len(points_hor):
                break
            block = Hough.cut_image(image, points_hor, start, 0)  # block为横条灰度，一条
            if block is None:
                continue
            _, thresh_block = cv2.threshold(block, 150, 255, cv2.THRESH_BINARY_INV)
            bar_list = cut.Cut(thresh_block, 0)  # bar_list是横切条列表，二值图像
            for bar in bar_list:
                piece_list = cut.cut_pieces(bar)
                for piece in piece_list:
                    enhanced_piece = enhance.upsampling_piece(piece)
                    ret, enhanced_piece_thresh = cv2.threshold(enhanced_piece, 130, 255, cv2.THRESH_BINARY_INV)
                    enhanced_piece_list.append(enhanced_piece_thresh)
                    self.output(out, self.ocr_recognize(enhanced_piece_thresh, 8))
                    # self.output(out, self.ocr_recognize(piece))
                    kk += 1
                    print(kk, 'in ', len(piece_list))
                # enhanced_list.append(enhanced_piece_list)
                piece_list_total.append(piece_list)
                k += 1
                print(k, 'in ', len(points_hor))

    # 整个图像横切成条，对条识别，再整合
    def line_recognize(self, image, out):
        cut = cut_lines()
        enhance = word_enhance()
        Hough = Hough_lines()
        text_list = []
        text_piece_list = []
        piece_list_total = []
        enhanced_list = []
        enhanced_piece_list = []
        points_hor = Hough.find_lines(image)
        k = 0
        bar_list = cut.Cut(image, 0)  # bar_list是横切条列表，二值图像
        for bar in bar_list:
            self.output(out, self.ocr_recognize(bar, 7))
        # return bar_list

    # 输出文本函数
    def output(self, out, text):
        result = open(out, 'w')
        result.write(text)
        result.close()

