from paddleocr import PaddleOCR, draw_ocr

class ocr_recognize:
    def ocr_paddle(self, image):
        # 无输入，输出为按目录加载的ocr模型
        # 模型目录：E:\PaddleOCR\PaddleOCR\inference
        # 修改模型目录文件：C:\Users\不居小杰\AppData\Roaming\Python\Python37\site-packages\paddleocr\tools\infer\utility.py
        # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
        # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
        ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
        # img_path = r'E:\文档\处方单识别\demo\screen.jpg'
        return ocr

    def ppocr_recognize(self, image, ocr_model):
        # image is img_path 输入为图片路径或图片数组(三通道RGB或灰度均可)
        # result = ocr.ocr(img_path, cls=True)
        result = ocr_model.ocr(image, cls=True)
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        # for line in result:
        #     print(line)

        # 显示结果
        # from PIL import Image
        #
        # image = Image.open(img_path).convert('RGB')
        # im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
        # im_show = Image.fromarray(im_show)
        # im_show.save('result.jpg')
        return boxes, txts, scores

    def output(self, text, out = 'E:\文档\处方单识别\project\output\\'+'result2'+'.txt'):
        result = open(out, 'a')
        result.write(text)
        result.write('\n')
        result.close()