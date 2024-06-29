from ultralytics import YOLOv10

print('Loading Model ')
checkpoint_path = "/Users/Girish/PycharmProjects/yolov10_kidneystone/yolov10_checkpoints/yolov10m.pt"
model = YOLOv10(checkpoint_path)
print('Finish Loading Model')

print('Starting Model Train')
model.train(data="/Users/Girish/PycharmProjects/yolov10_kidneystone/kidney_stone_data_roboflow/data.yaml",
            epochs=500,
            batch=64,
            imgsz=640)
print('Finished Model Train')