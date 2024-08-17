import numpy as np  
import matplotlib.pyplot as plt  
import os
import cv2

def read_flow_file(filename):    # 输入光流文件'.flo'，以向量的形式输表征光流特征
    with open(filename, 'rb') as f:  
        magic = np.fromfile(f, np.float32, count=1)  
        if 202021.25 != magic:  
            print("Magic number incorrect. Invalid .flo file")  
        else:  
            w = np.fromfile(f, np.int32, count=1).squeeze()  
            h = np.fromfile(f, np.int32, count=1).squeeze()  
            data = np.fromfile(f, np.float32, count=2*w*h)  
            flow = np.resize(data, (h, w, 2))  
    return flow   

def normalize_flow_in_region(flow, x_range, y_range):  # 计算了ROI中心点附近的光流整体流动方向和
    # 限定光流强度归一化的区域  
    flow_region = flow[int(y_range[0]):int(y_range[1]), int(x_range[0]):int(x_range[1]), :]  
    # 计算光流的模长  
    flow_magnitude = np.sqrt(flow_region[:,:,0]**2 + flow_region[:,:,1]**2)  
    total_magnitude = np.sum(flow_magnitude)  
    return total_magnitude  #normalized_flow  

def flow_sensor(flow,img,roi):
    perception_roi = [roi[0], roi[1], 2 * roi[2]]  
    init_roi = [710, 140, 64]  # (x, y, w/2)  
    step = 10  
    # 创建一个与输入图像大小相同的空白图像  
    canvas = np.copy(img)  
    # 获取ROI区域内的光流数据  
    roi_flow = flow[max(0, int(roi[1] - init_roi[2])):int(roi[1] + init_roi[2]), int(roi[0] - init_roi[2]):min(int(roi[0] + init_roi[2]), 1024), :]  
    # 计算ROI区域内光流的均值  
    mean_roi_flow = np.mean(roi_flow, axis=(0, 1)) 
    # 将均值光流的起始点设置为ROI区域的中心  
    arrow_start_x = roi[0]  
    arrow_start_y = roi[1]  
    # 绘制均值光流的箭头  
    cv2.arrowedLine(canvas, (int(arrow_start_x), int(arrow_start_y)), (int(arrow_start_x + 10*mean_roi_flow[0]), int(arrow_start_y + 0.001*mean_roi_flow[1])), (0, 255, 0),3)  
    # 绘制矩形框  
    cv2.rectangle(canvas, (int(roi[0] - roi[2]), int(roi[1] - roi[2])), (int(roi[0] + roi[2]), int(roi[1] + roi[2])), (255, 0, 0), 2)  
    # 其他部分的箭头  
    for y in range(0, flow.shape[0], step):  
        for x in range(0, flow.shape[1], step):  
            if (np.abs(flow[y, x, 0]) + np.abs(flow[y, x, 1])) < 1:  
                continue  
            cv2.arrowedLine(canvas, (x, y), (int(x + flow[y, x, 0]), int(y + flow[y, x, 1])), (0, 0, 255), 1)  
    # 计算绝对值之和  
    absolute_sum = np.abs(flow[:, :, 0]) + np.abs(flow[:, :, 1])  
    # 获取绝对值之和最大的前 5000 个值的索引  
    top_5000_indices = np.unravel_index(np.argsort(absolute_sum, axis=None)[-5000:], absolute_sum.shape) 
    mean_flow = np.array([0, 0], dtype=np.float64)  
    # 计算平均光流  
    for index in range(5000):  
        y, x = top_5000_indices[0][index], top_5000_indices[1][index]  
        mean_flow += flow[y, x]  
    mean_flow /= 5000  
    # 找到矩形框的边界  
    min_x = top_5000_indices[1].min()  
    max_x = top_5000_indices[1].max()  
    min_y = top_5000_indices[0].min()  
    max_y = top_5000_indices[0].max()  
    squre = (max_x-min_x)*(max_y-min_y)
    x_center = int((min_x+max_x)/2)
    y_center = int((min_y+max_y)/2)
    length = np.sqrt((roi[0]-x_center)**2+(roi[1]-y_center)**2)

    # roi update
    k_x , k_y=10,0.001
    new_roi = [roi[0]+min(k_x*(mean_roi_flow[0]),60),roi[1]+k_y*(mean_roi_flow[1]),roi[2]] 
    k_w = 1
    if squre<40000 and length<2*perception_roi[2]:
        k_w=1.8
    if squre<20000 and length<perception_roi[2]:
        k_w=0.9
    w =k_w*roi[2]
    w = min(w,218)
    w = max(w,64)
    new_roi = [roi[0]+min(k_x*(mean_roi_flow[0]),100),roi[1]+k_y*(mean_roi_flow[1]),w] 
    # 绘制矩形框和箭头表示平均光流  
    # canvas = np.zeros((436,1024, 3), dtype=np.uint8)  
    cv2.rectangle(canvas, (min_x, min_y), (max_x, max_y), (0, 0, 255), 1)  
    cv2.arrowedLine(canvas, (int((min_x + max_x) / 2), int((min_y + max_y) / 2)),   
                    (int((min_x + max_x) / 2) + int(mean_flow[0]), int((min_y + max_y) / 2) + int(mean_flow[1])),   
                    (0, 0, 255), 2)  
    
    # # 显示图像  
    # cv2.imshow("Result", canvas)  
    # cv2.waitKey(0)  
    # cv2.destroyAllWindows()  
    # # 保存绘制的图像  
    # cv2.imwrite('flow_visualization.jpg', canvas)  
      
    return canvas, new_roi  


def visualize_flow(flow, img, roi):    # 光流特征、原图、上一次的区域（更新为推测的ROI区域，而不是上一帧的ROI区域）
    perception_roi = [roi[0], roi[1], 2 * roi[2]]  
    init_roi = [710, 140, 64]  # (x, y, w/2)  
    step = 10  
    # 创建一个与输入图像大小相同的空白图像  
    canvas = np.copy(img)  
    # 获取ROI区域内的光流数据  
    roi_flow = flow[max(0, int(roi[1] - init_roi[2])):int(roi[1] + init_roi[2]), int(roi[0] - init_roi[2]):min(int(roi[0] + init_roi[2]), 1024), :]  
    # 计算ROI区域内光流的均值  
    mean_roi_flow = np.mean(roi_flow, axis=(0, 1)) 
    # 将均值光流的起始点设置为ROI区域的中心  
    arrow_start_x = roi[0]  
    arrow_start_y = roi[1]  
    # 绘制均值光流的箭头  
    cv2.arrowedLine(canvas, (int(arrow_start_x), int(arrow_start_y)), (int(arrow_start_x + 10*mean_roi_flow[0]), int(arrow_start_y + 0.001*mean_roi_flow[1])), (0, 255, 0),3)  
    # 绘制矩形框  
    cv2.rectangle(canvas, (int(roi[0] - roi[2]), int(roi[1] - roi[2])), (int(roi[0] + roi[2]), int(roi[1] + roi[2])), (255, 0, 0), 2)  
    # 其他部分的箭头  
    for y in range(0, flow.shape[0], step):  
        for x in range(0, flow.shape[1], step):  
            if (np.abs(flow[y, x, 0]) + np.abs(flow[y, x, 1])) < 1:  
                continue  
            cv2.arrowedLine(canvas, (x, y), (int(x + flow[y, x, 0]), int(y + flow[y, x, 1])), (0, 0, 255), 1)  
    # 计算绝对值之和  
    absolute_sum = np.abs(flow[:, :, 0]) + np.abs(flow[:, :, 1])  
    # 获取绝对值之和最大的前 5000 个值的索引  
    top_5000_indices = np.unravel_index(np.argsort(absolute_sum, axis=None)[-5000:], absolute_sum.shape) 
    mean_flow = np.array([0, 0], dtype=np.float64)  
    # 计算平均光流  
    for index in range(5000):  
        y, x = top_5000_indices[0][index], top_5000_indices[1][index]  
        mean_flow += flow[y, x]  
    mean_flow /= 5000  
    # 找到矩形框的边界  
    min_x = top_5000_indices[1].min()  
    max_x = top_5000_indices[1].max()  
    min_y = top_5000_indices[0].min()  
    max_y = top_5000_indices[0].max()  
    squre = (max_x-min_x)*(max_y-min_y)
    x_center = int((min_x+max_x)/2)
    y_center = int((min_y+max_y)/2)
    length = np.sqrt((roi[0]-x_center)**2+(roi[1]-y_center)**2)

    # roi update
    k_x , k_y=10,0.001
    new_roi = [roi[0]+min(k_x*(mean_roi_flow[0]),60),roi[1]+k_y*(mean_roi_flow[1]),roi[2]] 
    k_w = 1
    if squre<40000 and length<2*perception_roi[2]:
        k_w=1.8
    if squre<20000 and length<perception_roi[2]:
        k_w=0.9
    w =k_w*roi[2]
    w = min(w,218) # 436
    w = max(w,64)
    new_roi = [roi[0]+min(k_x*(mean_roi_flow[0]),100),roi[1]+k_y*(mean_roi_flow[1]),w] 
    # 绘制矩形框和箭头表示平均光流  
    # canvas = np.zeros((436,1024, 3), dtype=np.uint8)  
    cv2.rectangle(canvas, (min_x, min_y), (max_x, max_y), (0, 0, 255), 1)  
    cv2.arrowedLine(canvas, (int((min_x + max_x) / 2), int((min_y + max_y) / 2)),   
                    (int((min_x + max_x) / 2) + int(mean_flow[0]), int((min_y + max_y) / 2) + int(mean_flow[1])),   
                    (0, 0, 255), 2)  
    
    # # 显示图像  
    # cv2.imshow("Result", canvas)  
    # cv2.waitKey(0)  
    # cv2.destroyAllWindows()  
    # # 保存绘制的图像  
    # cv2.imwrite('flow_visualization.jpg', canvas)  
      
    return canvas, new_roi  


def increment_filename(filename):  
    prefix = filename[:-8]  # 提取前缀部分，比如 "frame_"  
    number = int(filename[-8:-4])  # 提取数字部分并转换为整数  
    extension = filename[-4:]  # 提取文件扩展名部分  
  
    # 增加数字部分并格式化为指定长度  
    new_number = str(number + 1).zfill(4)  
  
    # 组合新的文件名  
    new_filename = f"{prefix}{new_number}{extension}"  
  
    return new_filename  

if __name__ == "__main__":
    # 处理文件夹中的图片  
    input_folder = "/Users/tony/Desktop/project/optical_flow/crop/result9/inference/run.epoch-0-flow-field"    # 使用了裁剪后的图片进行处理，减少光流检测的噪声
    #input_folder = '/Users/tony/Desktop/project/optical_flow/video1/result3/inference/run.epoch-0-flow-field'  # 未裁剪的原图
    output_folder = "/Users/tony/Desktop/project/optical_flow/crop/result9/vis_jiantou_total_direct"  
    # 确保输出文件夹存在  
    if not os.path.exists(output_folder):  
        os.makedirs(output_folder)  
    flo_files = [filename for filename in os.listdir(input_folder) if filename.endswith(".flo")]       # 获取文件夹中所有的.flo文件  
    sorted_flo_files = sorted(flo_files)   # 按文件名排序  
    init_roi = [710,140,64]  #(x,y,w/2) 初始化ROI区域
    # 处理文件夹中的每张图片  ｜｜  每张图片单独处理
    for filename in sorted_flo_files:   #os.listdir(input_folder)
        if int(filename[2:6]) == 0: # 第一帧
            roi = init_roi
        if filename.endswith(".flo"):  
            flow_file = os.path.join(input_folder, filename)  
            flow = read_flow_file(flow_file)    # 提取光流特征，将.flo文件转换为可读的np数据  
            number = int(filename[2:6])         # 命名 
            filename_new = str(number + 1).zfill(4) 
            img = 'crop/result9/injection_crop_3/frame_'+filename_new+'.png'
            print(f'Successfully save cropped img! Path: {img}')  
            img = cv2.imread(img)               # ROI区域update+可视化
            vis_flow,new_roi = visualize_flow(flow,img,roi)  # ROI update + vis
            output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")  
            cv2.imwrite(output_file, vis_flow)  
            print("可视化后的光流图像已保存为", output_file)  
        roi = new_roi # roi update
    # break
# 示例用法  
# 假设你有一个光流的数组叫做 flow  
# visualize_flow(flow)  
