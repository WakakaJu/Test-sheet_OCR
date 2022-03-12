# Test-sheet_OCR
化验单ocr识别，目的识别出化验项目词条、结果数值及其定位</br>
**均为项目过程中的小demo，算是记录自己的实践经历，仅供学习与参考**

## 文件说明
`pretreatment.py`   是图片的预处理，包括将图片转化为灰度和二值(大津法OTSU)</br>
`cut_lines.py`   用于切割图片，将找到的项目内容进行横向和竖向切割</br>
`Hough_lines.py`   使用霍夫变换(Hough) 找出图片中的线，便于选出化验单图片中项目主体内容的部分，也包括沿线的横切竖切</br>
`word_enhance.py`   使用上采样的方法，对图像增强，令字图像更加清晰，便于识别</br></br>
`ocr.py`   为最先方案，首次使用谷歌tesseract-ocr识别引擎进行识别，效果不佳，后换为paddle引擎，此脚本废弃</br>
`ocr_paddle.py`   使用百度PaddleOCR识别引擎及官方infer模型进行识别</br>
`total_ppocr.py`   是整体识别和获取鼠标点击位置的类</br></br>
`ppocr.py`   使用了霍夫变换识别线找到项目内容，将项目主体切割成词组(字)，然后识别</br>
`content_ppocr.py`   也使用了霍夫，但仅仅按霍夫线将项目主体分栏分条切割，然后识别。识别结果用嵌套列表存储</br>
`printloc_word.py`   实现了整体识别后，在图像上点击字，通过获取鼠标点击位置把点击的字的识别结果打印出来的功能</br>
`main_paddle.py`   main文件，可以选择以上不同的识别方案，最终输出识别结果，可将识别结果进一步保存为txt文本文件。其中，`'print'`参数可实现与 `printloc_word.py` 文件相同的功能，即在图片上点击字，并把所点击的字的识别结果打印出来</br></br>
## 一些PaddleOCR引擎参数的说明
参数 `use_angle_cls` 指定是否使用方向分类器</br>
参数 `use_space_char` 指定是否预测空格字符</br>
参数 `--use_gpu=False` 指定是否使用GPU加速（与是否为Paddle是否为GPU版本挂钩）</br>
可视化识别结果默认保存到 `./inference_results` 文件夹里面。</br>
图片的路径可以是指定一个图片，也可以是指定一个文件夹中的所有图片。</br>
