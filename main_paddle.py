import cv2
import numpy as np

def run_one(image, kind = 'total'):
    '''
    :param image: 用于识别的化验单图片
    :param kind:
    'cut_block'是切成词组再识别
    'cut_line'是只按霍夫线切再识别
    'small'是用于识别项目数少的小单子
    'total'是整体识别
    'print'是在图像上点击字，然后把点击的字的识别结果打印出来，功能相当于printloc_word.py文件
    :return:
    '''
    if kind == 'cut_block':
        # ppocr运行
        from project.ppocr import ppocr
        ppocr = ppocr()
        text = ppocr.ppocr(image)

    elif kind == 'total':
        # main_paddle运行
        from paddleocr import PaddleOCR
        from project.word_enhance import word_enhance
        from project.ocr_paddle import ocr_recognize
        imPath = 'E:\文档\处方单识别\demo\screen.jpg'
        image = cv2.imdecode(np.fromfile(imPath, dtype=np.uint8), -1)
        enhance = word_enhance()
        ocr = ocr_recognize()
        en_img = enhance.upsampling_piece(image)
        ocr_model = PaddleOCR(use_angle_cls=True, lang="ch")
        _, text, _ = ocr.ppocr_recognize(en_img, ocr_model)

    elif kind == 'cut_line':
        # content_ppocr运行
        imPath = 'E:\文档\处方单识别\demo\screen.jpg'
        image = cv2.imdecode(np.fromfile(imPath, dtype=np.uint8), -1)
        from project.content_ppocr import content_pp
        con_pp = content_pp()
        text = con_pp.ppocr_content(image)

    elif kind == 'small':
        # 小单子处理uncomplated
        imPath = 'E:\文档\处方单识别\demo\screen2.jpg'
        image = cv2.imdecode(np.fromfile(imPath, dtype=np.uint8), -1)
        from project.ocr_paddle import ocr_recognize
        ocr = ocr_recognize()
        text = ppocr.ppocr(image)

    elif kind == 'print':
        # 点击出字
        from project.total_ppocr import total_pp, get_mouse
        imPath = 'E:\文档\处方单识别\demo\screen.jpg'
        image = cv2.imdecode(np.fromfile(imPath, dtype=np.uint8), -1)
        a, b = [], []
        total_pp = total_pp()
        mouse = get_mouse()
        boxes, text = total_pp.ppocr_total(image)
        def on_EVENT_LBUTTONDOWN(event, x, y,flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                xy = "%d,%d" % (x, y)
                a.append(x)
                b.append(y)
                cv2.circle(image, (x, y), 1, (0, 0, 255), thickness=-1)
                # cv2.putText(image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
                cv2.imshow("image", image)
        cv2.imshow("image", image)
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # print(a[-1], b[-1])
        for k in range(len(boxes)-1):
            if b[-1] > boxes[k][0][1] and b[-1] < boxes[k][3][1]:
                if a[-1] > boxes[k][0][0] and a[-1] < boxes[k][1][0]:
                    print(text[k])
        return text

if __name__ == '__main__':

    imPath = 'E:\文档\处方单识别\demo\screen.jpg'
    image = cv2.imdecode(np.fromfile(imPath, dtype=np.uint8), -1)
    text = run_one(image, kind = 'total')
