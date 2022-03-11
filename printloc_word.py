import cv2
import numpy as np
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