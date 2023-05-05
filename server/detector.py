import cv2
import numpy as np
import copy
from pathlib import Path

class Detector:
    def __init__(self, model_path, classes, imgsize, colors=None, score_thr=0.25, nms_thr=0.45) -> None:
        self.model = cv2.dnn.readNetFromONNX(model_path)
        # self.model = torch.jit.load(model_path)
        self.classes = classes
        if colors is None:
            colors = np.random.uniform(low=0.0,high=255.9,size=(len(classes), 3))
            colors = [(int(b), int(g), int(r)) for b, g, r in colors]
        self.colors = colors 
        self.imgsize = imgsize
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        
        
    def predict(self, original_image):
        model = self.model
        CLASSES = self.classes
        imgsize = self.imgsize
        
        [height, width, _] = original_image.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image
        scale = length / imgsize

        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(imgsize, imgsize))
        model.setInput(blob)
        outputs = model.forward()

        # outputs = model(torch.from_numpy(blob))
        # outputs = outputs.numpy()

        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= self.score_thr:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, self.score_thr, self.nms_thr, 0.5)

        detections = []
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                'class_id': class_ids[index],
                'class_name': CLASSES[class_ids[index]],
                'confidence': scores[index],
                'box': box,
                'scale': scale}
            detections.append(detection)
            
            self.draw_bounding_box(original_image, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                            round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))
        return detections, original_image

    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        CLASSES = self.classes
        colors = self.colors
        
        label = f'{CLASSES[class_id]} ({confidence:.2f})'
        color = colors[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



class CocoDetector(Detector):
    def __init__(self, model_path) -> None:
        super().__init__(
            model_path,
            classes=('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'), 
            colors=None,
            imgsize=640, 
            score_thr=0.25, 
            nms_thr=0.45
            )


def read_image(img_path, cv2_imread_flag=cv2.IMREAD_COLOR):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), flags=cv2_imread_flag)
    return img

def save_image(img, save_path):
    cv2.imencode(str(Path(save_path).suffix), img)[1].tofile(save_path)

if __name__ == '__main__':
    net_path = r"D:\code\inference_web\server\models\yolov8n.onnx"
    img_path = r"D:\code\inference_web\app\data\demo.jpg"

    detector = CocoDetector(net_path)
    img = read_image(img_path)
    detection, img_vis = detector.predict(copy.deepcopy(img))
    cv2.imshow("1", img_vis)
    cv2.waitKey()
    # save_path = f"{Path(img_path).stem}_result.jpg"
    # save_image(img_vis, save_path)

