import numpy as np  
import matplotlib.pyplot as plt  
import os
import cv2
import random  

def read_flow_file(filename):    # 输入光流文件'.flo'，以向量的形式输表征光流特征
    with open(filename, 'rb') as f:  
        magic = np.fromfile(f, np.float32, count=1)  
        # if 202021.25 != magic:  
        if magic.size > 0 and not np.isclose(202021.25, magic):
            print("Magic number incorrect. Invalid .flo file")  
        else:  
            w = np.fromfile(f, np.int32, count=1).squeeze()  
            h = np.fromfile(f, np.int32, count=1).squeeze()  
            # w = 1024
            # h = 436
            data = np.fromfile(f, np.float32, count=2*1024*436)  
            flow = np.resize(data, (h, w, 2))  
    return flow   

def add_gaussian_probability(set, x0, y0):  
    variance = 10.0  # 方差  
    probabilities = []
    for i in range(len(set)):
        distance = ((set[i][0] - x0) ** 2 + (set[i][1] - y0) ** 2) / (2 * variance)  # 计算欧氏距离 
        w = np.exp(-distance) / (2 * np.pi * variance)  # 计算权重概率
        probabilities.append(w)
    normalized_probabilities = probabilities / np.sum(probabilities)  
    for i in range(len(set)):
        set[i].append(normalized_probabilities[i])
    return set

def add_bivariate_gaussian_probability(set, x0, y0, x1, y1):  
    variance = 10.0  # 方差  
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

def monte_carlo_simulation(x0, y0, x1, y1):  
    particle_set = []  
    num = (int((x1-x0)/20)*int((y1-y0)/20))
    for i in range(num):  
        # 取上一次ROI的中央1/4区域
        x = random.randint(max(int((x0+x1)/2)-int((x1-x0)/8),300), min(int((x0+x1)/2)+int((x1-x0)/8),720))  # 
        y = random.randint(max(int((y0+y1)/2)-int((y1-y0)/8),130), min(int((y0+y1)/2)+int((y1-y0)/8),200))  
        # s =  max(np.arctan((-720+x)/105)*128016/(np.arctan(-4))+16384,4096*4)  # 使用了y = arctan(-x) 函数 （-4，0）的区间来进行映射 ｜ 用于匹配粒子处于不同位置时理想的面积
        s =  max(np.arctan((-720+x)/105)*173712/(np.arctan(-4))+16384,4096*4)
        particle_set.append([x, y, s])  
    return particle_set  

def find_covering_rectangle_for_multiple_squares(square_array):  
    min_x = float('inf')  
    min_y = float('inf')  
    max_x = float('-inf')  
    max_y = float('-inf')  
    # 遍历每个正方形，找到其四个顶点的坐标  
    for square in square_array:  
        x, y, s = square  # 分别获取正方形的信息  
        half_s = int((s)**0.5)/2  # 计算正方形边长的一半  
        # 找到正方形的四个顶点  
        vertices = [  
            (x - half_s, y - half_s),  
            (x + half_s, y - half_s),  
            (x - half_s, y + half_s),  
            (x + half_s, y + half_s)  
        ]  
        # 更新最小外接矩形的坐标  
        min_x = int(min(min_x, min(v[0] for v in vertices))) 
        min_y = int(min(min_y, min(v[1] for v in vertices)))  
        max_x = int(max(max_x, max(v[0] for v in vertices))) 
        max_y = int(max(max_y, max(v[1] for v in vertices)))
    # 返回最小外接矩形的左上角和右下角坐标  
    return min_x, min_y,max_x, max_y

def Particle_Maker(roi_set,count=2):
    particle_set = [] # 存放粒子，粒子的表示方法和ROI一致
    if len(roi_set) == 1 : # roi_0 可以符合高斯分布来构建粒子
        x,y,w=roi_set[0][0],roi_set[0][1],int(roi_set[0][2]**0.5)
        x0,y0,x1,y1=x-int(w/2),y-int(w/2),x+int(w/2),y+int(int(w/2))
        particle_set = monte_carlo_simulation(x0,y0,x1,y1)  # 依据蒙特卡洛方法构建粒子集合
        particle_set = add_gaussian_probability(particle_set,x,y)  # 依据二维高斯分布为粒子设置归一化初始权重
    else: # normal situation
        x0,y0,x1,y1 = find_covering_rectangle_for_multiple_squares(roi_set[(len(roi_set)-count):len(roi_set)])  # 得到前两个ROI的并集区域
        particle_set = monte_carlo_simulation(x0,y0,x1,y1) # 依据蒙特卡洛方法构建粒子集合
        c1_x,c1_y = roi_set[len(roi_set)-1][0],roi_set[len(roi_set)-1][1]
        c2_x,c2_y = roi_set[len(roi_set)-2][0],roi_set[len(roi_set)-2][1]
        particle_set = add_bivariate_gaussian_probability(particle_set,c1_x,c1_y,c2_x,c2_y)  # 依据二维高斯分布为粒子设置归一化初始权重
    return particle_set

def calculate_intensity_in_squares(set, flow):  
    intensities = []  
    for i in range(len(set)):  
        x, y, s = set[i][0],set[i][1],set[i][2]
        side_length = int(((s)**0.5)/2)
        # 提取正方形区域内的光流强度  
        square_flow = flow[(y-side_length):(y+side_length), (x-side_length):(x+side_length), :]  
        # 计算光强均值  
        mean_intensity = np.mean(square_flow, axis=(0, 1))  
        mean_intensity = np.linalg.norm(mean_intensity)
        intensities.append(mean_intensity)  
    nor_intensities = intensities / np.sum(intensities)
    for i in range(len(set)):
        set[i].append(nor_intensities[i])  # 添加光强分布概率值w_i set = [[x,y,s,w,w_i],...]
    return set

def flow_init(flow):
    num = int(flow.shape[0]*flow.shape[1]*0.5)
    # 计算绝对值之和  
    absolute_sum = np.abs(flow[:, :, 0]) + np.abs(flow[:, :, 1])  
    # 获取绝对值之和最大的前 5000 个值的索引  
    top_num_indices = np.unravel_index(np.argsort(absolute_sum, axis=None)[-num:], absolute_sum.shape) 
    mean_flow = np.array([0, 0], dtype=np.float64)  
    # 计算平均光流  
    for index in range(num):  
        y, x = top_num_indices[0][index], top_num_indices[1][index]  
        mean_flow += flow[y, x]  
    mean_flow /= num  
    # 找到矩形框的边界  
    min_x = top_num_indices[1].min()  
    max_x = top_num_indices[1].max()  
    min_y = top_num_indices[0].min()  
    max_y = top_num_indices[0].max()  
    squre = (max_x-min_x)*(max_y-min_y)
    x_center = int((min_x+max_x)/2)
    y_center = int((min_y+max_y)/2)
    return x_center,y_center,squre,mean_flow

def calculate_direction(roi,flow):
    x, y, s = roi[0],roi[1],roi[2]
    side_length = int(((s)**0.5)*1.5/2) # perception
    # 针对感受野中的区域进行flow 的过滤
    square_flow = flow[(y-side_length):(y+side_length), (x-side_length):(x+side_length), :] 
    x_center,y_center,squre,mean_flow = flow_init(flow)
    mean_direction = np.mean(square_flow, axis=(0, 1)) 
    mean_direction = mean_flow
    return mean_direction

def direct_gaussian_simulation(particle_set,direct,roi_set):  # diretion = [y,x]
    c_x, c_y = roi_set[len(roi_set)-1][0]+direct[0]*20,roi_set[len(roi_set)-1][1]+direct[1]*0.0000001
    # print(f'direct[0]:{direct[0]};direct[1]={direct[1]}')
    variance = 10.0  # 方差  
    probabilities = []
    for i in range(len(particle_set)):
        distance = ((particle_set[i][0] - c_x) ** 2 + (particle_set[i][1] - c_y) ** 2) / (2 * variance)  # 计算欧氏距离 
        w = np.exp(-distance) / (2 * np.pi * variance)  # 计算权重概率
        probabilities.append(w)
    normalized_probabilities = probabilities / np.sum(probabilities)  
    for i in range(len(particle_set)):
        particle_set[i].append(normalized_probabilities[i])
    return particle_set

def weight_average(set,weight):
    for i in range(len(set)):
        w_0,w_1,w_2 = set[i][3],set[i][4],set[i][5]
        w_sum = w_0 * weight[0]+w_1*weight[1]+w_2*weight[2]
        set[i].append(w_sum)
    return set

def Flow_Sensor(flow_file, particle_set, roi_set): # 输入光流文件，更新粒子中的权重set=[[x,y,s,w],[x,y,s,w],...]
    flow = read_flow_file(flow_file)    # 提取光流特征，将.flo文件转换为可读的np数据    flow.shape = 384;1024
    particle_set = calculate_intensity_in_squares(particle_set,flow) # 更新计算粒子区域的平均光强，添加基于光强的概率分布值
    
    perception_roi = [roi_set[len(roi_set)-1][0],roi_set[len(roi_set)-1][1],roi_set[len(roi_set)-1][2]*2]  # perception_roi 在原ROI基础上提高2倍
    direction = calculate_direction(perception_roi,flow)
    # print(f'direction = {direction}')
    particle_set = direct_gaussian_simulation(particle_set,direction,roi_set) # [[x,y,s,w_0,w_1,w_2],[],...]
    weight = [0.2,0.3,0.5]
    particle_set = weight_average(particle_set,weight)
    # print(f'set = {set}')
    return particle_set   # [[x,y,s,w_0,w_1,w_2,w_3],[x,y,s,w_0,w_1,w_2,w_3],...]

def roi_update(set):
    max_w = 0
    max_i = 0
    for i in range(len(set)):
        if set[i][6]>max_w:
            max_w = set[i][6]
            max_i = i
    roi = set[max_i][0:3]
    return roi

def visualize(img,flow_file,roi):
    flow = read_flow_file(flow_file)    # 提取光流特征，将.flo文件转换为可读的np数据    flow.shape = 384;1024
    img = cv2.imread(img)
    # 创建一个与输入图像大小相同的空白图像  
    canvas = np.copy(img) 
    side_length = int(((roi[2])**0.5)/2)
    cv2.rectangle(canvas, (int(roi[0] - side_length), int(roi[1] - side_length)), (int(roi[0] + side_length), int(roi[1] + side_length)), (255, 0, 0), 2)  
    return canvas

if __name__ == "__main__":
    # 处理文件夹中的图片  
    input_folder = "/Users/tony/Desktop/project/optical_flow/crop/result7/inference/run.epoch-0-flow-field"    # 使用了裁剪后的图片进行处理，减少光流检测的噪声
    #input_folder = '/Users/tony/Desktop/project/optical_flow/video1/result3/inference/run.epoch-0-flow-field'  # 未裁剪的原图
    output_folder = "/Users/tony/Desktop/project/optical_flow/crop/result7/particle_filter"  
    # 确保输出文件夹存在  
    if not os.path.exists(output_folder):  
        os.makedirs(output_folder)  
    flo_files = [filename for filename in os.listdir(input_folder) if filename.endswith(".flo")]       # 获取文件夹中所有的.flo文件  
    sorted_flo_files = sorted(flo_files)   # 按文件名排序  
    roi_0 = [710,140,4096*4]  #(x,y,s) 初始化ROI区域
    roi_set = [roi_0]
    # 处理文件夹中的每张图片  ｜｜  每张图片单独处理
    for filename in sorted_flo_files:   #os.listdir(input_folder)
        if filename.endswith(".flo"):  # 读取文件中的光流文件，并找到对应的图片
            flow_file = os.path.join(input_folder, filename)  
            number = int(filename[2:6])         # 命名 
            filename_new = str(number + 1).zfill(4) 
            img = 'crop/result7/injection_cropped/frame_'+filename_new+'.png'
            print(f'Successfully save cropped img! Path: {img}')  # 得到裁剪后的图像
            Particle_Set = Particle_Maker(roi_set,count=2) # 针对之前的ROI时间序列信息，重新构建粒子
            Particle_Set = Flow_Sensor(flow_file,Particle_Set,roi_set) # 根据光流信息，更新粒子的概率值
            roi_new = roi_update(Particle_Set)  # 根据粒子概率值，得到新的roi范围
            roi_set.append(roi_new)  # 添加新的roi范围到roi序列中
            # Visualize
            vis_img = visualize(img,flow_file,roi_new)
            output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png") 
            cv2.imwrite(output_file, vis_img) 
            print("可视化后的光流图像已保存为", output_file) 
        # if len(roi_set)==1:
        #     break

        # if int(filename[2:6]) == 0: # 第一帧
        #     roi = roi_0
        #     Particle_Maker(roi_set,count=2)
        # if filename.endswith(".flo"):  # 正式处理：完成粒子集合的生成--权重的初始化—-FlowNet观测—-权重更新-- ROI迭代
        #     flow_file = os.path.join(input_folder, filename)  
        #     number = int(filename[2:6])         # 命名 
        #     filename_new = str(number + 1).zfill(4) 
        #     img = 'crop/result7/injection_cropped/frame_'+filename_new+'.png'
        #     print(f'Successfully save cropped img! Path: {img}')  
        #     img = cv2.imread(img)               # ROI区域update+可视化

        #     # flow = read_flow_file(flow_file)    # 提取光流特征，将.flo文件转换为可读的np数据  
        #     vis_flow,new_roi = visualize_flow(flow,img,roi)  # ROI update + vis
 
        # roi = new_roi # roi update
        # roi_set.append(roi)


    # break
# 示例用法  
# 假设你有一个光流的数组叫做 flow  
# visualize_flow(flow)  
