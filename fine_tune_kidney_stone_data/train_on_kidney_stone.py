from ultralytics import YOLOv10
import os
root_path = "" # deinfe your working directory path

print('Loading Model ')
checkpoint_path = os.path.join(root_path,"yolov10_kidneystone/yolov10_checkpoints/yolov10m.pt")
model = YOLOv10(checkpoint_path)
print('Finish Loading Model')

print('Starting Model Train')
data_yaml_path = os.path.join(root_path,"yolov10_kidneystone/kidney_stone_data_roboflow/data.yaml")
model.train(data=data_yaml_path,
            epochs=500,
            batch=64,
            imgsz=640)#,single_cls=True)
print('Finished Model Train')