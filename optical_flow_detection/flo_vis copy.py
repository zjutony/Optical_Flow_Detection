import numpy as np  
import matplotlib.pyplot as plt  
import os
import cv2

def read_flow_file(filename):  
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
# def visualize_flow(flow, img):  
#     fig, ax = plt.subplots()  
#     ax.imshow(img)  
#     step = 10  
      
#     for y in range(0, flow.shape[0], step):  
#         for x in range(0, flow.shape[1], step):  
#             ax.arrow(x, y, flow[y, x, 0], flow[y, x, 1], color='r', head_width=5, head_length=5)  
      
#     fig.canvas.draw()  
#     width, height = fig.canvas.get_width_height()  
#     data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))[:, :, :3]  
  
#     plt.close(fig)  # 关闭图形界面  
  
#     return data  
def visualize_flow(flow, img, roi):  
    perception_roi = [roi[0],roi[1],2*roi[2]]
    fig, ax = plt.subplots()  
    ax.imshow(img)  
    step = 10  
      
    for y in range(0, flow.shape[0], step):  
        for x in range(0, flow.shape[1], step):  
            ax.arrow(x, y, flow[y, x, 0], flow[y, x, 1], color='r', head_width=2, head_length=3)  
            #print(f"x=({x}),y=({y}),flow=({flow[y, x, 0]},{flow[y, x, 0]})")
    
    # 计算绝对值之和  
    absolute_sum = np.abs(flow[:, :, 0]) + np.abs(flow[:, :, 1])  
    # 获取绝对值之和最大的前 10000 个值的索引  
    top_10000_indices = np.unravel_index(np.argsort(absolute_sum, axis=None)[-10000:], absolute_sum.shape)  
    mean_flow_x,mean_flow_y = 0,0
    # 打印这些值  
    for index in range(10000):  
        y, x = top_10000_indices[0][index], top_10000_indices[1][index]  
        print(f"Index {index+1}: Absolute sum: {absolute_sum[y, x]}, Flow values: ({flow[y, x, 0]}, {flow[y, x, 1]}) at (x={x}, y={y})")  
        mean_flow_x = mean_flow_x+flow[y,x,0]
        mean_flow_y = mean_flow_y+flow[y,x,1]
    # 找到矩形框的边界  
    min_x = top_10000_indices[1].min()  
    max_x = top_10000_indices[1].max()  
    min_y = top_10000_indices[0].min()  
    max_y = top_10000_indices[0].max()      
    # 绘制矩形框  
    rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, linewidth=1, edgecolor='r', facecolor='none')  
    ax.add_patch(rect)  
    # 绘制箭头表示平均光流  
    x_center = int((min_x+max_x)/2)
    y_center = int((min_y+max_y)/2)
    ax.arrow(x_center, y_center, int(mean_flow_x/10000), int(mean_flow_y/10000), color='b', head_width=5)  
    # plt.show()
    fig.canvas.draw()  
    data = np.array(fig.canvas.renderer.buffer_rgba())  
  
    plt.close(fig)  # 关闭图形界面  
  
    return data  

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
# 处理文件夹中的图片  
input_folder = "/Users/tony/Desktop/project/optical_flow/crop/result7/inference/run.epoch-0-flow-field"  
#input_folder = '/Users/tony/Desktop/project/optical_flow/video1/result3/inference/run.epoch-0-flow-field'
output_folder = "/Users/tony/Desktop/project/optical_flow/crop/result7/vis_jiantou_total_direct"  
  
# 确保输出文件夹存在  
if not os.path.exists(output_folder):  
    os.makedirs(output_folder)  

# 获取文件夹中所有的.flo文件  
flo_files = [filename for filename in os.listdir(input_folder) if filename.endswith(".flo")]  
# 按文件名排序  
sorted_flo_files = sorted(flo_files)  

init_roi = [710,140,128]  #(x,y,w)

# 处理文件夹中的每张图片  
for filename in sorted_flo_files:   #os.listdir(input_folder)
    if int(filename[2:6]) == 0:
        roi = init_roi
    if filename.endswith(".flo"):  
        flow_file = os.path.join(input_folder, filename)  
        flow = read_flow_file(flow_file)  
        number = int(filename[2:6])
        filename_new = str(number + 1).zfill(4) 
        img = 'crop/result7/injection_crop_1/frame_'+filename_new+'.png'
        print(img)
        img = cv2.imread(img)
        vis_flow,new_roi = visualize_flow(flow,img,roi)  
        output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")  
        cv2.imwrite(output_file, vis_flow)  
        print("可视化后的光流图像已保存为", output_file)  
    roi = new_roi # roi update
    # break
# 示例用法  
# 假设你有一个光流的数组叫做 flow  
# visualize_flow(flow)  