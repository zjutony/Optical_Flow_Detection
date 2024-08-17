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


def increment_filename(filename):  
    prefix = filename[:-8]  # 提取前缀部分，比如 "frame_"  
    number = int(filename[-8:-4])  # 提取数字部分并转换为整数  
    extension = filename[-4:]  # 提取文件扩展名部分  
  
    # 增加数字部分并格式化为指定长度  
    new_number = str(number + 1).zfill(4)  
  
    # 组合新的文件名  
    new_filename = f"{prefix}{new_number}{extension}"  
  
    return new_filename  
  
# # 测试  
# filename1 = "frame_0000.png"  
# filename2 = "frame_0150.png"  
# filename3 = "frame_0003.png"  
  
# print(increment_filename(filename1))  # 输出: frame_0001.png  
# print(increment_filename(filename2))  # 输出: frame_0151.png  
# print(increment_filename(filename3))  # 输出: frame_0004.png  
# def visualize_flow(flow,img):  
#     fig, ax = plt.subplots()  
#     ax.imshow(img)  
#     # 可调参数，用于控制箭头的密度  
#     step = 10   
      
#     # 在图像上绘制箭头  
#     for y in range(0, flow.shape[0], step):  
#         for x in range(0, flow.shape[1], step):  
#             ax.arrow(x, y, flow[y, x, 0], flow[y, x, 1], color='r', head_width=5, head_length=5)  
#     # 将图像转换为数组  
#     fig.canvas.draw()  
#     data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  
#     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))  
  
#     plt.close(fig)  # 关闭图形界面  
  
#     return data  
    # plt.show()  

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


# def visualize_flow(flow, img, roi):  
#     perception_roi = [roi[0],roi[1],2*roi[2]]
#     init_roi = [710,140,64]  #(x,y,w/2)
#     fig, ax = plt.subplots()  
#     ax.imshow(img)  
#     step = 10  
    
#         # 获取ROI区域内的光流数据  
#     roi_flow = flow[max(0,int(roi[1]-init_roi[2])):int(roi[1]+init_roi[2]), int(roi[0]-init_roi[2]):min(int(roi[0]+init_roi[2]),1024), :]  
#     # 计算ROI区域内光流的均值  
#     mean_roi_flow = np.mean(roi_flow, axis=(0, 1))  
#     # 将均值光流的起始点设置为ROI区域的中心  
#     arrow_start_x = roi[0]  
#     arrow_start_y = roi[1] 
#     # 绘制均值光流的箭头  
#     ax.arrow(arrow_start_x, arrow_start_y, mean_roi_flow[0], mean_roi_flow[1], color='g', head_width=5)  
#     # 绘制矩形框  
#     rect_roi = plt.Rectangle((roi[0]-roi[2], roi[1]-roi[2]), 2*roi[2], 2*roi[2], linewidth=1, edgecolor='b', facecolor='none') 
#     ax.add_patch(rect_roi)  
#     # roi update
#     k_x , k_y=10,0.001
#     new_roi = [roi[0]+min(k_x*(mean_roi_flow[0]),60),roi[1]+k_y*(mean_roi_flow[1]),roi[2]] 

#     # roi_weight = normalize_flow_in_region(flow,x_range=(roi[0]-roi[2],roi[0]+roi[2]),y_range=(roi[1]-roi[2],roi[1]+roi[2]))
#     # perception_roi_weight = normalize_flow_in_region(flow,x_range=((perception_roi[0]-perception_roi[2],perception_roi[0]+perception_roi[2])),y_range=((perception_roi[1]-perception_roi[2],perception_roi[1]+perception_roi[2])))
#     # k_w = np.sqrt(perception_roi_weight/roi_weight)
#     # if k_w<1:
#     #     w=(0.1*k_w)*roi[2]+0.9*roi[2]
#     # else:
#     #     w=(0.3*k_w)*roi[2]+0.7*roi[2]
#     # w = min(w,218)
#     # w = max(w,64)
#     # new_roi = [roi[0]+min(k_x*(mean_roi_flow[0]),100),roi[1]+k_y*(mean_roi_flow[1]),w] 
#     for y in range(0, flow.shape[0], step):  
#         for x in range(0, flow.shape[1], step):  
#             if (np.abs(flow[y, x, 0]) + np.abs(flow[y, x, 1]))<1:
#                 continue
#             ax.arrow(x, y, flow[y, x, 0], flow[y, x, 1], color='r', head_width=2, head_length=3)  
#             # print(f"x=({x}),y=({y}),flow=({flow[y, x, 0]},{flow[y, x, 0]})")
    
#     # 计算绝对值之和  
#     absolute_sum = np.abs(flow[:, :, 0]) + np.abs(flow[:, :, 1])  
#     # 获取绝对值之和最大的前 5000 个值的索引  
#     top_5000_indices = np.unravel_index(np.argsort(absolute_sum, axis=None)[-5000:], absolute_sum.shape)  
#     mean_flow_x,mean_flow_y = 0,0
    
#     # 打印这些值  
#     for index in range(5000):  
#         y, x = top_5000_indices[0][index], top_5000_indices[1][index]  
#         print(f"Index {index+1}: Absolute sum: {absolute_sum[y, x]}, Flow values: ({flow[y, x, 0]}, {flow[y, x, 1]}) at (x={x}, y={y})")  
        
#         mean_flow_x = mean_flow_x+flow[y,x,0]
#         mean_flow_y = mean_flow_y+flow[y,x,1]
#     mean_flow_x = int(mean_flow_x/5000)
#     mean_flow_y = int(mean_flow_y/5000)
#     # 找到矩形框的边界  
#     min_x = top_5000_indices[1].min()  
#     max_x = top_5000_indices[1].max()  
#     min_y = top_5000_indices[0].min()  
#     max_y = top_5000_indices[0].max()      
#     squre = (max_x-min_x)*(max_y-min_y)
#     x_center = int((min_x+max_x)/2)
#     y_center = int((min_y+max_y)/2)
#     length = np.sqrt((roi[0]-x_center)**2+(roi[1]-y_center)**2)
#     k_w = 1
#     if squre<40000 and length<2*perception_roi[2]:
#         k_w=1.8
#     if squre<20000 and length<perception_roi[2]:
#         k_w=0.9
#     w =k_w*roi[2]
#     w = min(w,218)
#     w = max(w,64)
#     new_roi = [roi[0]+min(k_x*(mean_roi_flow[0]),100),roi[1]+k_y*(mean_roi_flow[1]),w] 
#     # 绘制矩形框  
#     rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, linewidth=1, edgecolor='r', facecolor='none')  
#     ax.add_patch(rect)  
#     # 绘制箭头表示平均光流  
#     ax.arrow(x_center, y_center, (mean_flow_x/5000), (mean_flow_y/5000), color='b', head_width=5)  
#     # plt.show()
#     fig.canvas.draw()  
#     data = np.array(fig.canvas.renderer.buffer_rgba())  
  
#     plt.close(fig)  # 关闭图形界面  
  
#     return data,new_roi