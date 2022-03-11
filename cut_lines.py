from project.pretreatment import pretreat
import cv2
import numpy as np
class cut_lines:
    # 统计每行黑点个数，参数为二值图数组
    def countPoint(self, img):

        re = []
        for th in img:
            re.append(sum(th) / 255)
        return re


    # 根据每行黑点个数选取进行切割的位置，参数为二值图数组，以及切割方向
    def findPoint(self, img, axis):
        if (axis == 1):
            img = img.T
        cut_thresh = 0.05 * img.shape[axis] * (1 - axis) + 0.05 * img.shape[axis] * axis
        start = -1
        end = -1
        result = []
        countx = self.countPoint(img)
        for x in range(len(countx)):
            if (countx[x] > cut_thresh and start < 0):
                start = x - 1
                end = x + 1
            elif (countx[x] > cut_thresh):
                end = x + 1
            elif (countx[x] <= cut_thresh and start > 0):
                result.append([start, end])
                start, end = -1, -1
        return result

    # 根据切割位置进行分割，参数为二值图数组，以及切割方向
    def Cut(self, img, axis):
        pre = pretreat()
        if img.ndim == 3:    # 输入为三通道彩色原图
            _, thresh = pre.binary(img)
        else:                # 输入为二值图像
            thresh = img
        point = self.findPoint(thresh, axis)
        re = []
        for x in point:
            if (axis == 0):
                re.append(thresh[x[0]: x[1],:])
            elif (axis == 1):
                re.append(thresh[:, x[0]: x[1]])
        return re

    # 切条成字
    def cut_pieces(self, strip):
        piece = self.Cut(strip,1)
        return piece

    # 把横切的小条再切分成小块，形成一个一个的词组、短语，以长条高度为阈值判断
    # 找到小块坐标
    def find_block(self, bar):
        count = np.sum(bar, axis = 0) / 255
        start, end = [], []
        no_zeros = np.where(count != 0)
        start.append(no_zeros[0][0])
        if len(no_zeros[0]) > 0:
            for i in range(len(no_zeros[0])):
                if i+1 >= len(no_zeros[0]):
                    break
                if no_zeros[0][i+1]-no_zeros[0][i]>= bar.shape[0]:
                    start.append(no_zeros[0][i+1])
                    end.append(no_zeros[0][i])
        end.append(no_zeros[0][-1])
        return start, end

    # 把bar切成小块
    def cut_block(self, bar):
        block_list = []
        start, end = self.find_block(bar)
        for j in range(len(start)):
            if j+1 >= len(start):
                break
            block = bar[:, start[j]: end[j]]
            block_list.append(block)
        return block_list

    # 找到切出来的块
    # def find_location(self, location, content_part, ):
    #     part =


if __name__ == '__main__':
    cut = cut_lines()
    imPath = 'E:\文档\处方单识别\demo\screen.jpg'
    # imPath = 'E:\文档\处方单识别\data\sheets\hyd1.png'
    image = cv2.imdecode(np.fromfile(imPath, dtype=np.uint8), -1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bar_list = cut.Cut(image, 0)
    piece_list = []
    for bar in bar_list:
        piece_list.append(cut.cut_pieces(bar))
    # block = []
    #
    # block = [part for part in block if part is not None]
    # vtitch = np.vstack((block[i] for i in range(len(block))))
    # cv2.imshow('cut_result', vtitch)
    cv2.imshow('bar', bar_list[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()