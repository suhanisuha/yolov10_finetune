from ultralytics import YOLOv10
import os


if __name__=='__main__':
    root_path = "" # deinfe your working directory path

    img_path = os.path.join(root_path ,"yolov10_kidneystone/fine_tune_kidney_stone_data/images/Bhawarth_in_Hall.png")
    img_path = os.path.join(root_path ,"yolov10_kidneystone/fine_tune_kidney_stone_data/images/Modi_with_Gary.png")

    pretrained_checkpoint_path = os.path.join(root_path , "yolov10_kidneystone/fine_tune_kidney_stone_data/pretrained_checkpoints/yolov10m.pt")
    model = YOLOv10(pretrained_checkpoint_path)
    print('Finish Loading Model')

    print('Doing Prediction on a Natural Image')
    result = model.predict(img_path)
    print('Finish Prediction on Image')

    print('Show Results ')
    import matplotlib.pyplot as plt
    plt.imshow(result[0].plot())
    plt.savefig(img_path.split(".")[0]+'_labelled.png')
    plt.show()
    print('Done')
