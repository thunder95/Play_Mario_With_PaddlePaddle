import math
import numpy as np

# 计算三点的夹角
# 相对于point_2
def cal_ang(point_1, point_2, point_3):
    """
    根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 返回任意角的夹角值，这里只是返回点2的夹角
    """
    a = math.sqrt(
        (point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (point_2[1] - point_3[1]))
    b = math.sqrt(
        (point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (point_1[1] - point_3[1]))
    c = math.sqrt(
        (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (point_1[1] - point_2[1]))
    # A = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
    B = 0
    if a != 0 and c !=0:
        B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
    # C = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))
    return B


# 计算两条线是否相交
def cross_point(line1, line2):
    point_is_exist = False
    x = y = 0
    x1,y1,x2,y2 = line1
    x3,y3,x4,y4 = line2

    if (x2 - x1) == 0:
        k1 = None
        b1 = 0
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键

    if (x4 - x3) == 0:  # L2直线斜率不存在
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k1 is None:
        if not k2 is None:
            x = x1
            y = k2 * x1 + b2
            point_is_exist = True
    elif k2 is None:
        x = x3
        y = k1 * x3 + b1
    elif not k2 == k1:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        point_is_exist = True

    return point_is_exist, [x, y]


#得到向量的坐标以及向量的模长
class Point(object):
    def __init__(self, x):
        self.x1 = x[0]
        self.y1 = x[1]
        self.x2 = x[2]
        self.y2 = x[3]
    def vector(self):
        c = (self.x1 - self.x2, self.y1 - self.y2)
        return c
    def length(self):
        d = math.sqrt(pow((self.x1 - self.x2), 2) + pow((self.y1 - self.y2), 2))
        return d
#计算向量夹角
class Calculate(object):
    def __init__(self, x, y, m, n):
        self.x = x
        self.y = y
        self.m = m
        self.n = n
    def Vector_multiplication(self):
        self.mu = np.dot(self.x, self.y)
        return self.mu
    def Vector_model(self):
        self.de = self.m * self.n
        return self.de
    def cal(self):
        result = Calculate.Vector_multiplication(self) / Calculate.Vector_model(self)
        return result