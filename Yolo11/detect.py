import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO('../runs/train5/weights/best.pt')
    model.predict(source='images',
                imgsz=640,
                device='0',
                save=True
                )

