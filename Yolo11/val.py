import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO('../runs/train5/weights/best.pt')
    model.val(data=r'C:\Users\YAOYAO\Desktop\毕业论文\训练\YOLOv11\data.yaml',
                imgsz=640,
                batch=16,
                split='test',
                workers=0,
                device='cpu',
                )
 
