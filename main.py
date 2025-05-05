import sensor, image, time, lcd
import os
import KPU as kpu
from board import board_info
from Maix import GPIO,FFT
from fpioa_manager import fm
import utime
import gc
import openmv_numpy as np1
from kalman_filter import Tracker_Manager
from perspective_transformation import PerspectiveTransformation
"""
其他函数
"""
def get_max_value_objects(objects):#如果有同一类有多个对象，返回概率最大的那一个
    max_objects = {}
    for obj in objects:
        classid = obj.classid()
        if classid not in max_objects or obj.value() > max_objects[classid].value():
            max_objects[classid] = obj
    return list(max_objects.values())
#得出各个物体框的坐标,将swimmingpool和swimmer的坐标分别存入两个列表中
def get_objects_coordinate(objects):
    swimmer_coords = []
    for obj in objects:
        swimmer_coords.append(rect_to_coords(obj.rect()))
    return swimmer_coords
#将rect转换为边界框四个点坐标
def rect_to_coords(rect):
    x, y, w, h = rect
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
#将边界框四个点坐标转换为rect
def coords_to_rect(coords):
    x1, y1 = coords[0]
    x2, y2 = coords[2]
    w = x2 - x1
    h = y2 - y1
    return (x1, y1, w, h)
#得出游泳者的中心位置
def get_swimmer_center(swimmer_coords):
    centers = []
    for swimmer in swimmer_coords:
        x1, y1 = swimmer[0]
        x2, y2 = swimmer[2]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        centers.append((center_x, center_y))
    return centers
#得出swimmer在swimmingpool的相对位置
def get_relative_position(swimmingpool_coords, swimmer_center):
    relative_positions = 0
    for swimmingpool in swimmingpool_coords:
        x1, y1 = swimmingpool[0] #取第一个游泳池的坐标
        x2, y2 = swimmingpool[2]
    for swimmer in swimmer_center:
        x,y = swimmer_center[0] #取第一个游泳者的坐标
    relative_positions=(y-y1)/(y2-y1)
    return relative_positions
start_label = 0
fps=0
labels = ['swimmer']
img_swimmingpool_coord=[[50,10],[180,10],[220,200],[20,200]]
real_swimmingpool_coord=[[0,0],[0,100],[200,100],[0,200]]
swimmer_coord=[]
predicted_swimmer_coord=[]
rectangle_width=0
rectangle_height=0
#摄像头初始化
sensor_window = (224, 224)
sensor_hmirror=False
sensor_vflip=False
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing(sensor_window)
sensor.set_hmirror(sensor_hmirror)
sensor.set_vflip(sensor_vflip)
sensor.run(1)
#lcd初始化
lcd_rotation=0
lcd.init(type=1)
lcd.rotation(lcd_rotation)
lcd.clear(lcd.WHITE)
#时钟对象初始化
clock = time.clock()                # 创建一个时钟对象来跟踪FPS。
# 透视矩阵初始化，计算透视变换矩阵
H = PerspectiveTransformation.compute_homography(img_swimmingpool_coord, real_swimmingpool_coord)
#卡尔曼滤波器初始化
A = np1.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])#A: 状态转移矩阵
H_k = np1.eye(4)#H: 观测矩阵
Q = np1.eye(4, value=0.1)#  # Q: 过程噪声协方差矩阵
R = np1.eye(4)# R: 测量噪声协方差矩阵
B = None
Manager = Tracker_Manager()
#蓝色的LAB阈值(0, 100, -25, 18, -44, -11)

anchors = [1.78, 1.31, 3.88, 2.34, 2.69, 1.16, 1.22, 0.66, 0.56, 0.59]

task=kpu.load("/sd/model-191671.kmodel")  # 模型保存在SD卡中，从SD卡中直接加载模型
#kpu.load_kmodel(0x300000,278440)  # 我们需要把kmodel模型烧入到0x300000的位置，278440为模型的大小，我们可以通过查看文件属性可以得到；
kpu.init_yolo2(task,0.5,0.3,len(anchors)//2,anchors)
"""
kpu_net: kpu 网络对象, 即加载的模型对象, KPU.load()的返回值
threshold: 概率阈值， 只有是这个物体的概率大于这个值才会输出结果， 取值范围：[0, 1]
nms_value: box_iou 门限, 为了防止同一个物体被框出多个框，当在同一个物体上框出了两个框，这两个框的交叉区域占两个框总占用面积的比例 如果小于这个值时， 就取其中概率最大的一个框
anchor_num: anchor 的锚点数， 这里固定为 len(anchors)//2
anchor: 锚点参数与模型参数一致，同一个模型这个参数是固定的，和模型绑定的（训练模型时即确定了）， 不能改成其它值。
"""


while True:
    clock.tick()  # 更新FPS时钟
#    print("mem free:",gc.mem_free())  # 查询剩余内存
    if fps <= 0 or fps > 1000:  # 检查FPS是否有效
        fps = 10  # 设置默认FPS
#    print("fps:", fps)  # 打印FPS

    img=sensor.snapshot()
    objects=kpu.run_yolo2(task, img)#objects形式[{"x":41, "y":57, "w":139, "h":171, "value":0.832165, "classid":0, "index":0, "objnum":3}, {"x":83, "y":62, "w":57, "h":44, "value":0.611361, "classid":1, "index":1, "objnum":3}, {"x":127, "y":87, "w":46, "h":44, "value":0.471382, "classid":1, "index":2, "objnum":3}]
#    print(objects)
    img.draw_line(img_swimmingpool_coord[0][0], img_swimmingpool_coord[0][1], img_swimmingpool_coord[1][0], img_swimmingpool_coord[1][1], color=(255, 0, 0), thickness=2)
    img.draw_line(img_swimmingpool_coord[1][0], img_swimmingpool_coord[1][1], img_swimmingpool_coord[2][0], img_swimmingpool_coord[2][1], color=(255, 0, 0), thickness=2)
    img.draw_line(img_swimmingpool_coord[2][0], img_swimmingpool_coord[2][1], img_swimmingpool_coord[3][0], img_swimmingpool_coord[3][1], color=(255, 0, 0), thickness=2)
    img.draw_line(img_swimmingpool_coord[3][0], img_swimmingpool_coord[3][1], img_swimmingpool_coord[0][0], img_swimmingpool_coord[0][1], color=(255, 0, 0), thickness=2)
    if objects:
        max_objects = get_max_value_objects(objects)
        swimmer_coords = get_objects_coordinate(max_objects)
        if swimmer_coords:
            swimmer_coord = swimmer_coords
        if swimmer_coord:
            print("Swimmer coordinates:", swimmer_coord)
            swimmer_center = get_swimmer_center(swimmer_coord)
            print("Swimmer center:", swimmer_center)
            Manager.match(int(swimmer_center[0][0]),int(swimmer_center[0][1]),A,H_k,Q,R,20,20)
            transformed = PerspectiveTransformation.apply_homography(H, *swimmer_center[0])
            print("Transformed coordinates:", transformed)
        for object in max_objects:#用于显示每个物体的概率和类别
            rect = object.rect()
            rectangle_height = rect[3]
            rectangle_width = rect[2]
            img.draw_rectangle(rect, color=(255, 0, 0))
            img.draw_string(rect[0], rect[1], "%s:%.2f" % (labels[object.classid()], object.value()), scale=2, color=(255, 0, 0))
    else:
        predicted_swimmer_counter=0
        img.draw_string(112, 112, "none", scale=2, color=(255, 0, 0))
    Manager.update()
    trails_pre = Manager.get_motion_trail_pre()
    #在没有检测到物体的情况下，显示预测的轨迹
    if not objects:
        for ID, trail in trails_pre:
            if len(trail):
                x, y = trail[len(trail)-1][0]-rectangle_width*0.5, trail[len(trail)-1][1]-rectangle_height*0.5
                img.draw_rectangle(int(x), int(y), rectangle_width,rectangle_height, color=(0, 255, 0))
                img.draw_string(int(x), int(y), "predict_trail", scale=2, color=(0, 255, 0))
    print("Trails:", trails_pre)
    lcd.display(img)
    fps = clock.fps()  # 获取FPS
    gc.collect()      # 内存回收机制

kpu.deinit()
