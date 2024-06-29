from ultralytics import YOLOv10
import os


if __name__=='__main__':
    root_path = "" # deinfe your working directory path

    data_yaml_path = os.path.join(root_path, "yolov10_finetune/fine_tune_kidney_stone_data/kidney_stone_data_roboflow/data.yaml")

    pretrained_checkpoint_path = os.path.join(root_path,"yolov10_finetune/fine_tune_kidney_stone_data/pretrained_checkpoints/yolov10m.pt")
    print('Loading Pre Trained YOLO Model ')
    model = YOLOv10(pretrained_checkpoint_path)
    print('Finish Loading Model')

    print('Starting Fine Tune Train on Kideny Stone Dataset')

    model.train(data=data_yaml_path,
                epochs=500,
                batch=64,
                imgsz=640)

    print('Finished Model Train')