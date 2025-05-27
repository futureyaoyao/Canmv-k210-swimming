# -*- coding:utf-8 -*-
# 文件名: predict.py
# 功能: 使用 YOLO 模型进行目标检测，支持图片和视频的检测与可视化

import cv2  # OpenCV库，用于图像处理
from ultralytics import YOLO  # YOLO库，用于加载和运行YOLO模型
import os  # 操作系统相关功能，如路径操作
import argparse  # 命令行参数解析
import time  # 时间相关操作，用于计算FPS
import torch  # PyTorch库，用于深度学习模型的加载和推理
import numpy as np  # NumPy库，用于数值计算

# 创建命令行参数解析器
parser = argparse.ArgumentParser()
# 添加检测参数
parser.add_argument('--weights', default=r"runs/detect/train3/weights/best.pt", type=str, help='weights path')  # 模型权重路径
parser.add_argument('--source', default=r"videos/input3.mp4", type=str, help='img or video(.mp4)path')  # 输入文件路径
#parser.add_argument('--source', default=r"images", type=str, help='img or video(.mp4)path')  # 输入文件路径
parser.add_argument('--save', default=r"save", type=str, help='save img or video path')  # 输出保存路径
parser.add_argument('--vis', default=True, action='store_true', help='visualize image')  # 是否可视化检测结果
parser.add_argument('--conf_thre', type=float, default=0.2, help='conf_thre')  # 置信度阈值
parser.add_argument('--iou_thre', type=float, default=0.6, help='iou_thre')  # IOU阈值
opt = parser.parse_args()  # 解析命令行参数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 检测是否有GPU可用

if not os.path.exists(opt.save):
    os.makedirs(opt.save)

def get_color(idx):
    """
    根据索引生成颜色，用于绘制检测框
    :param idx: 索引值
    :return: 颜色 (B, G, R)
    """
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

class Detector(object):
    """
    检测器类，用于加载模型并执行目标检测，并支持整图透视变换
    """
    def __init__(self, weight_path, conf_threshold=0.5, iou_threshold=0.5,
                 src_points=None, dst_points=None):
        """
        初始化检测器
        :param weight_path: 模型权重路径
        :param conf_threshold: 置信度阈值
        :param iou_threshold: IOU阈值
        :param src_points: 透视变换源点（4个点）
        :param dst_points: 透视变换目标点（4个点）
        """
        self.device = device  # 设备 (CPU 或 GPU)
        self.model = YOLO(weight_path)  # 加载YOLO模型
        self.conf_threshold = conf_threshold  # 置信度阈值
        self.iou_threshold = iou_threshold  # IOU阈值
        self.names = self.model.names  # 类别名称
        # 透视变换矩阵
        self.M = None
        self.src_points = src_points  # 保存原始点用于绘制
        if src_points is not None and dst_points is not None:
            src = np.array(src_points, dtype=np.float32)
            dst = np.array(dst_points, dtype=np.float32)
            self.M = cv2.getPerspectiveTransform(src, dst)

    def perspective_transform(self, img):
        """
        对整张图片进行透视变换
        """
        if self.M is not None:
            h, w = img.shape[:2]
            # 输出尺寸可根据dst_points调整，这里用原图尺寸
            return cv2.warpPerspective(img, self.M, (w, h))
        return img

    def draw_points_and_lines(self, img, points):
        """
        在图像上绘制四个点及其连线
        :param img: 输入图片
        :param points: 点的列表 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        :return: 绘制后的图片
        """
        color_point = (0, 255, 255)  # 黄色
        color_line = (255, 0, 0)     # 蓝色
        thickness = 10
        # 绘制点
        for idx, (x, y) in enumerate(points):
            cv2.circle(img, (int(x), int(y)), 6, color_point, -1)
            cv2.putText(img, f'{idx+1}:({int(x)},{int(y)})', (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_point, 2)
        # 绘制连线
        for i in range(len(points)):
            pt1 = (int(points[i][0]), int(points[i][1]))
            pt2 = (int(points[(i+1)%len(points)][0]), int(points[(i+1)%len(points)][1]))
            cv2.line(img, pt1, pt2, color_line, thickness)
        return img

    def detect_image(self, img_bgr):
        """
        对单张图片进行目标检测，先检测再绘制点和连线，最后做透视变换
        :param img_bgr: 输入图片 (BGR 格式)
        :return: 检测后的图片
        """
        # 1. 目标检测
        results = self.model(img_bgr, verbose=True, conf=self.conf_threshold,
                             iou=self.iou_threshold, device=self.device)

        bboxes_cls = results[0].boxes.cls  # 检测框的类别
        bboxes_conf = results[0].boxes.conf  # 检测框的置信度
        bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')  # 检测框的坐标

        for idx in range(len(bboxes_cls)):
            box_cls = int(bboxes_cls[idx])  # 类别索引
            bbox_xyxy = bboxes_xyxy[idx]  # 检测框坐标
            #bbox_label = self.names[box_cls]  # 类别名称
            bbox_label = "swimmer" # 类别名称
            box_conf = f"{bboxes_conf[idx]:.2f}"  # 置信度
            xmax, ymax, xmin, ymin = bbox_xyxy[2], bbox_xyxy[3], bbox_xyxy[0], bbox_xyxy[1]

            # 绘制检测框和标签
            img_bgr = cv2.rectangle(
                img_bgr, (int(xmin), int(ymin)), (int(xmax), int(ymax)), get_color(box_cls + 3), 2)
            cv2.putText(
                img_bgr, f'{str(bbox_label)}/{str(box_conf)}', (int(xmin), int(ymin) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_color(box_cls + 3), 2)

        # 2. 绘制四个点和连线（以初始化时的src_points为例）
        # if self.M is not None:
        #     # 注意：src_points是初始化时传入的
        #     img_bgr = self.draw_points_and_lines(img_bgr, self.src_points)

        # 3. 透视变换
        #img_bgr = self.perspective_transform(img_bgr)
        return img_bgr

# 主程序入口
if __name__ == '__main__':
    # 透视变换点对（请根据实际场景填写4个点）
    src_points = [(100, 250), (500, 250), (550, 650), (0, 700)]
    dst_points = [(0, 0), (500, 0), (500, 1000), (0, 1000)]
    # 初始化检测器
    model = Detector(weight_path=opt.weights, conf_threshold=opt.conf_thre, iou_threshold=opt.iou_thre,
                     src_points=src_points, dst_points=dst_points)
    images_format = ['.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.JPEG']  # 支持的图片格式
    video_format = ['mov', 'MOV', 'mp4', 'MP4']  # 支持的视频格式

    # 如果输入是图片
    if os.path.join(opt.source).split(".")[-1] not in video_format:
        # 获取所有图片文件
        image_names = [name for name in os.listdir(opt.source) for item in images_format if
                       os.path.splitext(name)[1] == item]
        for img_name in image_names:
            img_path = os.path.join(opt.source, img_name)  # 图片路径
            img_ori = cv2.imread(img_path)  # 读取图片
            img_vis = model.detect_image(img_ori)  # 检测图片
            img_vis = cv2.resize(img_vis, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST)  # 调整大小
            cv2.imwrite(os.path.join(opt.save, img_name), img_vis)  # 保存检测结果

            if opt.vis:  # 如果需要可视化
                cv2.imshow(img_name, img_vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # 如果输入是视频
    else:
        capture = cv2.VideoCapture(opt.source)  # 打开视频文件
        fps = capture.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 获取视频尺寸
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 视频编码格式
        outVideo = cv2.VideoWriter(os.path.join(opt.save, os.path.basename(opt.source).split('.')[-2] + "_out.mp4"),
                                   fourcc,
                                   fps, size)  # 初始化视频写入对象

        total_frames = 0
        detected_frames = 0

        while True:
            ret, frame = capture.read()  # 读取视频帧
            if not ret:
                break
            total_frames += 1
            start_frame_time = time.perf_counter()  # 开始计时
            img_vis = model.detect_image(frame)  # 检测视频帧
            end_frame_time = time.perf_counter()  # 结束计时

            # 判断本帧是否检测到目标
            results = model.model(frame, verbose=False, conf=model.conf_threshold, iou=model.iou_threshold, device=model.device)
            bboxes_cls = results[0].boxes.cls
            if len(bboxes_cls) > 0:
                detected_frames += 1

            # 计算每帧的FPS
            elapsed_time = end_frame_time - start_frame_time
            fps_estimation = 1 / elapsed_time if elapsed_time > 0 else 0.0

            # 在视频帧上显示FPS
            h, w, c = img_vis.shape
            cv2.putText(img_vis, f"FPS: {fps_estimation:.2f}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2)

            outVideo.write(img_vis)  # 写入视频帧
            cv2.imshow('detect', img_vis)  # 显示视频帧
            cv2.waitKey(1)

        capture.release()  # 释放视频捕获对象
        outVideo.release()  # 释放视频写入对象

        print(f"视频总帧数: {total_frames}")
        print(f"检测到目标的帧数: {detected_frames}")