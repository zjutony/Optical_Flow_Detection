import cv2  
import os  
  
# 文件夹A的路径  
folder_path = 'video1/result3/injection_1'  
  
# 读取文件夹中的所有PNG图片  
images = [img for img in os.listdir(folder_path) if img.endswith(".png")]  
  
# 按文件名排序  
images.sort(key=lambda x: int(x.split('.')[0][6:10])) 
  
# 设置视频的帧率和分辨率  
frame_rate = 30  # 每秒1帧  
frame_width, frame_height = 1024, 436  # 可根据需要调整  
  
# 创建视频写入对象  
video_name = 'org_video_flow1.mp4'  
video_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (frame_width, frame_height))  
  
# 将图片逐帧写入视频  
for image in images:  
    image_path = os.path.join(folder_path, image)  
    frame = cv2.imread(image_path)  
    frame = cv2.resize(frame, (frame_width, frame_height))  # 调整图像大小以匹配视频分辨率  
    video_writer.write(frame)  
  
# 释放资源  
video_writer.release()  