# import the necessary packages
import numpy as np
import cv2


def load_yolo():
    """
    Load and create YOLO net
    :return:
    """
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def load_image(img_path):
    """
    Loading image by path
    :param img_path: path to image
    :return:
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels


def detect_objects(img, net, outputLayers):
    """
    Detect objects in the given img using the given net
    :return:
    """
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    """
    calculate and return box dimensions
    """
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    cv2.imshow("Image", img)


def image_detect(img_path):
    model, classes, colors, output_layers = load_yolo()
    image, height, width, channels = load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    add_text = detect_dog_on_sofa(boxes, class_ids)
    if add_text:
        new_frame = addText(image, 'Lily On Sofa!')
    else:
        new_frame = image
    draw_labels(boxes, confs, colors, class_ids, classes, new_frame)
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break


def start_video(video_path, out_path):
    """
    Main function - search for dof on sofa or ned and alaram about that!!
    :param video_path: path to video
    :return:
    """
    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)
    video_frames = []
    size = None
    while True:
        _, frame = cap.read()
        try:
            frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
        except Exception as e: break
        height, width, channels = frame.shape
        size = (width, height)
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        add_text = detect_dog_on_sofa(boxes, class_ids)
        new_frame = addText(frame, 'Lily On Sofa!') if add_text else frame
        draw_labels(boxes, confs, colors, class_ids, classes, new_frame)
        video_frames.append(np.copy(new_frame))
        key = cv2.waitKey(1)
        if key == 27: break

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    for im in video_frames:
        # writing for an images array
        out.write(im)
    out.release()
    cap.release()


def detect_dog_on_sofa(boxes, class_ids):
    """
    Detect dogs on sofa, given boxes of objects
    :return:
    """
    dog = 16
    sofa = 57
    chair = 56
    idx_array = np.array(class_ids)
    dog_indexes = np.where(idx_array == dog)[0]
    sofa_indexes = np.where(idx_array == sofa)[0]
    chair_indexes = np.where(idx_array == chair)[0]
    disallowed_indexes = np.concatenate([sofa_indexes, chair_indexes])
    for disallowed_index in disallowed_indexes:
        x0, y0, w0, h0 = boxes[disallowed_index]
        sofa_rect = get_rects_dict(h0, w0, x0, y0)
        for dog_index in dog_indexes:
            x1, y1, w1, h1 = boxes[dog_index]
            dog_rect = get_rects_dict(h1, w1, x1, y1)
            if are_intersect(sofa_rect, dog_rect):
                return True
    return False


def get_rects_dict(h0, w0, x0, y0):
    return {
        'top_left':
            {
                'x': x0,
                'y': y0
            },
        'bot_right':
            {
                'x': x0 + w0,
                'y': y0 + h0
            }
    }


def are_intersect(rect_a, rect_b):
    """
    The function checks if the rect are intersect and return results
    :param rect_a: rect of one object
    :param rect_b: rect of second object
    :return:
    """
    # If one rectangle is on left side of other

    a_before_b = rect_b['top_left']['x'] >= rect_a['bot_right']['x']
    b_before_a = rect_a['top_left']['x'] >= rect_b['bot_right']['x']

    if a_before_b or b_before_a:
        return False

    a_above_b = rect_b['top_left']['y'] >= rect_a['bot_right']['y']
    b_above_a = rect_a['top_left']['y'] >= rect_b['bot_right']['y']

    # If one rectangle is above other
    if a_above_b or b_above_a:
        return False

    return True


def addText(frame, text):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (50, 50)

    # fontScale
    font_scale = 1

    # Blue color in BGR
    color = (0, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    # set the text start position
    text_offset_x = org[0]
    text_offset_y = org[1]

    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, 0), (text_offset_x + text_width + 2, 100))

    rectangle_bgr = (255, 255, 255)

    cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)

    # Using cv2.putText() method
    new_frame = cv2.putText(frame, text, org, font,
                            font_scale, color, thickness, cv2.LINE_AA)

    return new_frame


if __name__ == '__main__':
    vid_path = 'c:\\temp\\dogs\\lily4.mp4'
    out_path = 'c:\\temp\\dogs\\lily_detect.mp4'
    start_video(vid_path, out_path)
