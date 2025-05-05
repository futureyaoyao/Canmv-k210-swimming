from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO(r'yolo11.yaml')  # 此处以 m 为例，只需写yolov11m即可定位到m模型
    model.train(data=r'C:\Users\YAOYAO\Desktop\毕业论文\训练\YOLOv11\data.yaml',
                imgsz=640,
                epochs=100,
                single_cls=True,  # 多类别设置False
                batch=16,
                workers=10,
                device='cpu',
                )
