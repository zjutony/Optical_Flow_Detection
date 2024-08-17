# import random  
# import matplotlib.pyplot as plt  
  
# def monte_carlo_simulation(x0, y0, x1, y1, num):  
#     particle_set = []  
#     num = (int((x1-x0)/10)*int((y1-y0)/10))
#     for i in range(num):  
#         x = random.randint(x0, x1)  
#         y = random.randint(y0, y1)  
#         particle_set.append((x, y))  
#     return particle_set  
  
# # 测试函数  
# x0, y0, x1, y1 = 0, 0, 256, 256  # 例如给定的矩形空间  
# num = 400  # 生成的点的数量  
# result = monte_carlo_simulation(x0, y0, x1, y1, num)  
  
# # 呈现生成的点  
# x, y = zip(*result)  # 将点的坐标分离  
# plt.scatter(x, y)  # 使用散点图呈现  
# plt.xlim(x0-1, x1+1)  # 设置x轴范围  
# plt.ylim(y0-1, y1+1)  # 设置y轴范围  
# plt.show()  


# import numpy as np  
# import matplotlib.pyplot as plt  
# from mpl_toolkits.mplot3d import Axes3D  
  
# # 生成两个二维高斯分布数据  
# x = np.linspace(-5, 5, 100)  
# y = np.linspace(-5, 5, 100)  
# x, y = np.meshgrid(x, y)  
  
# # 第一个高斯分布  
# z1 = np.exp(-(x**2 + y**2))  
  
# # 第二个高斯分布  
# z2 = np.exp(-((x-1)**2 + (y-1)**2))  
  
# # 将两个高斯分布相加得到重叠  
# z_combined = z1 + z2  
  
# # 绘制重叠的二维高斯分布图  
# fig = plt.figure()  
# ax = fig.add_subplot(111, projection='3d')  
# ax.plot_surface(x, y, z_combined, cmap='viridis')  
  
# # 设置图像标题和坐标轴标签  
# ax.set_title('Overlapped 2D Gaussian Distributions')  
# ax.set_xlabel('X')  
# ax.set_ylabel('Y')  
# ax.set_zlabel('Probability Density')  
  
# # 显示图像  
# plt.show()  
# import math
# import numpy as np  
# import matplotlib.pyplot as plt  
  
# # x = np.linspace(-10, 10, 1000)  
# # y = np.arctan(-x)  

# # plt.plot(x, y, label='y=tan(-x)')  
# # plt.xlabel('x')  
# # plt.ylabel('y')  
# # plt.title('Graph')  
# # plt.grid(True)  
# # plt.legend()  
# # plt.show()  
# x = 700
# s = np.arctan((-720+x)/105)*128016/(np.arctan(-4))+16384
# print(np.arctan((-720+x)/105))
# print(s)
# w = int(s**0.5)
# print(f'x = {x};w = {w}')


import random
import numpy as np  
import cv2  

# def add_gaussian_probability(x,y, x0, y0):  
#     variance = 10.0  # 方差  
#         # 计算欧氏距离  
#     distance = ((x - x0) ** 2 + (y - y0) ** 2) / (2 * variance)  
#     # 计算权重概率  
#     w = np.exp(-distance) / (2 * np.pi * variance)  
#     return w
def add_bivariate_gaussian_probability(set, x0, y0, x1, y1):  
    variance = 5.0  # 方差  
    probabilities = []
    for i in range(len(set)):
        distance0 = ((set[i][0] - x0) ** 2 + (set[i][1] - y0) ** 2) / (2 * variance)  # 计算欧氏距离 
        distance1 = ((set[i][0] - x1) ** 2 + (set[i][1] - y1) ** 2) / (2 * variance)  
        w = np.exp(-distance0) / (2 * np.pi * variance) + np.exp(-distance1) / (2 * np.pi * variance)  # 计算权重概率
        probabilities.append(w)
    normalized_probabilities = probabilities / np.sum(probabilities)  
    for i in range(len(set)):
        set[i].append(normalized_probabilities[i])
    return set
# 创建一个图像
img_path = 'crop/result7/injection_cropped/frame_0001.png'
img = cv2.imread(img_path)
img = np.array(img) 

def monte_carlo_simulation(x0, y0, x1, y1):  
    particle_set = []  
    num = (int((x1-x0)/20)*int((y1-y0)/20))
    # probabilities = []
    for i in range(num):  
        # 取上一次ROI的中央1/4区域
        x = random.randint(int((x0+x1)/2)-int((x1-x0)/8), int((x0+x1)/2)+int((x1-x0)/8))  # 
        y = random.randint(int((y0+y1)/2)-int((y1-y0)/8), int((y0+y1)/2)+int((y1-y0)/8))  
        s =  np.arctan((-720+x)/105)*128016/(np.arctan(-4))+16384  # 使用了y = arctan(-x) 函数 （-4，0）的区间来进行映射 ｜ 用于匹配粒子处于不同位置时理想的面积
        # probabilities.append(add_gaussian_probability(x,y,int((x0+x1)/2),int((y0+y1)/2)))
        particle_set.append([x, y, s])  
    # 归一化概率值  
    # normalized_probabilities = probabilities / np.sum(probabilities)  
    # for i in range(num):
    #     particle_set[i].append(normalized_probabilities[i])
    return particle_set  

# 生成随机的 n 个正方形信息  
roi_0 = [710,140,4096*4]  # 4096
set = [roi_0]
x,y,w=set[0][0],set[0][1],int(set[0][2]**0.5)
x0,y0,x1,y1=x-int(w/2),y-int(w/2),x+int(w/2),y+int(int(w/2))
particle_set = monte_carlo_simulation(x0, y0, x1, y1)  # 存储每个正方形的信息  
n = len(particle_set)
particle_set = add_bivariate_gaussian_probability(particle_set,x,y,x,y+10)
print(particle_set)

# 绘制每个正方形  
for partical in particle_set:  
    x, y, s, w = partical  
    side_length = int(np.sqrt(s) / 2)  
    cv2.rectangle(img, (x - side_length, y - side_length), (x + side_length, y + side_length), (0, 0, 255), 1) 
    if x>710 and y<140:
        cv2.putText(img, str(round(w,4)), (x+ side_length+int(0.1*x), y - side_length-int(0.1*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)  
    elif x<710 and y<140:
        cv2.putText(img, str(round(w,4)), (x- side_length-int(0.1*x), y - side_length-int(0.1*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)  
    elif x<710 and y>140:
        cv2.putText(img, str(round(w,4)), (x- side_length-int(0.1*x), y + side_length+int(0.1*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)  
    else:
        cv2.putText(img, str(round(w,4)), (x+ side_length+int(0.1*x), y + side_length+int(0.1*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)  
    
# 保存图像到路径 C  
cv2.imwrite('test.png', img)  



