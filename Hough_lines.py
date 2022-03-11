import cv2
import numpy as np
from project.pretreatment import pretreat
class Hough_lines:
    def line_detection(self, image):
        pre = pretreat()
        threshold_k = 0.28  # 以该系数*图像宽度再取整作为阈值
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3:
            image, _ = pre.binary(image)
        edges = cv2.Canny(image, 50, 50, apertureSize=3)

        # cv2.HoughLines()返回值是(ρ,θ),ρ的单位是像素，θ的单位是弧度
        # 输入的第一个参数为二值化图像，因此在做霍夫变换前需要预处理(Canny检测或其他)将图像变成二值化
        # 第二和第三个参数分别代表ρ和θ的精确度，第四个参数是阈值，只有累加其中的值高于阈值时才被认为是直线
        # 也可以把它看成 能检测到的直线的最短长度(以像素点为单位)

        print(image.shape, int(threshold_k * image.shape[1]))
        lines = cv2.HoughLines(edges, 1.0, np.pi / 180, int(threshold_k * image.shape[1]))
        lines = np.reshape(lines, (-1, 2))
        return lines


    def find_lines(self, image):
        lines = self.line_detection(image)
        # 输出横线竖线的端点坐标列表
        points_ver = []
        points_hor = []
        for line in lines:
            rho, theta = line
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            if (theta < np.pi / 4.0 or theta > 3.0 * np.pi / 4.0):
                # 画竖线
                point_ver1 = (int(rho / cos_theta), 0)
                point_ver2 = (int((rho / cos_theta - image.shape[0] * sin_theta) / cos_theta), image.shape[0])
                points_ver.append([point_ver1[0], 0, point_ver2[0], image.shape[0]])
            else:
                # 画横线
                point_hor1 = (0, int(rho / sin_theta))
                point_hor2 = (image.shape[1], int(rho / sin_theta - image.shape[1] * cos_theta / sin_theta))
                points_hor.append([0, point_hor1[1], image.shape[1], point_hor2[1]])
        points_hor.sort(key=lambda x: float(x[1]))
        points_ver.sort(key=lambda x: float(x[0]))
        return points_hor, points_ver


    def cut_image(self, image,points_list, start, axis):
        # 输出的是灰度图像, 一横条
        pre = pretreat()
        gray, thresh = pre.binary(image)
        # points_hor, points_ver = self.find_lines(image)
        if axis == 0:
            # 横切
            points_hor = points_list
            pos_from = min(points_hor[start][1], points_hor[start][3])
            pos_to = max(points_hor[start + 1][1], points_hor[start + 1][3])
            blank = np.zeros((pos_to - pos_from, gray.shape[1]), dtype=np.uint8)

            if points_hor[start + 1][1] - points_hor[start][1] >= 10:
                points_consent = np.where(gray[pos_from:pos_to, :] >= 50)
                blank[points_consent] = gray[points_consent[0] + pos_from, points_consent[1]]
                return blank
        elif axis == 1:
            # 竖切
            points_ver = points_list
            pos_from = min(points_ver[start][0], points_ver[start][2])
            pos_to = max(points_ver[start + 1][0], points_ver[start+1][2])
            blank = np.zeros((gray.shape[0], pos_to - pos_from), dtype=np.uint8)

            if points_ver[start + 1][0] - points_ver[start][0] >= 10:
                points_consent = np.where(gray[:, pos_from:pos_to] >= 50)
                blank[points_consent] = gray[points_consent[0],  pos_from + points_consent[1]]
                return blank


# if __name__ == '__main__':
#     Hough = Hough_lines()
#     imPath = 'E:\文档\处方单识别\demo\screen.jpg'
#     # imPath = 'E:\文档\处方单识别\data\sheets\hyd1.png'
#     image = cv2.imdecode(np.fromfile(imPath, dtype=np.uint8), -1)
#     lines = Hough.line_detection(image)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     points_hor, points_ver = Hough.find_lines(gray)
#     block = []
#     for start in range(len(points_hor)):
#         if start +1 >= len(points_hor)-1:
#             break
#         block.append(Hough.cut_image(image, start))
#     block = [part for part in block if part is not None]
#     vtitch = np.vstack((block[i] for i in range(len(block))))
#     cv2.imshow('cut_result', vtitch)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
