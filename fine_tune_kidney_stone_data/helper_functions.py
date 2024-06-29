
def get_preidction_rect_list(prediction_result):
    rect_list = []
    p = prediction_result[0].boxes.xywh.detach().cpu().numpy()
    from matplotlib import patches
    for item in p:
        rect = patches.Rectangle((item[0]-item[2]/2, item[1]-item[3]/2), item[2], item[3], linewidth=1, edgecolor='r', facecolor='none')
        rect_list.append(rect)

    return rect_list

def get_ground_truth_rect_list(label_path,img_shape):

    with open(label_path, 'r') as file:
        labels = file.readlines()

    image_height, image_width, _ = img_shape
    from matplotlib import patches
    rect_list = []
    for label in labels:
        class_id, x_center, y_center, width, height = map(float, label.strip().split())
        class_name = ''

        # Convert YOLO format to bounding box coordinates
        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height

        x_min = x_center - width / 2
        y_min = y_center - height / 2

        # Create a Rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='g', facecolor='none')
        rect_list.append(rect)

    return rect_list
