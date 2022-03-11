from project.pretreatment import pretreat
from paddleocr import PaddleOCR
from project.ocr_paddle import ocr_recognize
import cv2

class total_pp:
    def ppocr_total(self, image):
        pre = pretreat()  # 实例化
        ocr = ocr_recognize()

        # 载入模型
        ocr_model = PaddleOCR(use_angle_cls=True, lang="ch")

        _, thresh = pre.binary(image)
        boxes, text, scores = ocr.ppocr_recognize(image, ocr_model)
        return boxes, text

class get_mouse:
    def on_EVENT_LBUTTONDOWN(self, event, x, y,flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            a.append(x)
            b.append(y)
            cv2.circle(image, (x, y), 1, (0, 0, 255), thickness=-1)
            cv2.putText(image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image", image)