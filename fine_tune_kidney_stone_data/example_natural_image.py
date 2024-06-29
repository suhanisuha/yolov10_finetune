from ultralytics import YOLOv10

print('Loading Model ')
checkpoint_path = "/Users/Girish/PycharmProjects/yolov10_kidneystone/yolov10_checkpoints/yolov10m.pt"
model = YOLOv10(checkpoint_path)
print('Finish Loading Model')


print('Doing Prediction on a Natural Image')
img_path = "/Users/Girish/PycharmProjects/yolov10_kidneystone/fine_tune_kidney_stone_data/Modi_with_Gary.png"
img_path = "/fine_tune_kidney_stone_data/Bhawarth_in_Hall.png"
result = model.predict(img_path)
print('Finish Prediction on Image')

print('Show Results ')
import matplotlib.pyplot as plt
plt.imshow(result[0].plot())
plt.show()
print('Done')