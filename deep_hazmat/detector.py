import cv2
import time
import numpy as np

from deep_hazmat.segmentation import Segmentation
from deep_hazmat.common import Object
from deep_hazmat.nms import non_max_suppression

color_list = [
    (220, 120, 50),  # poison
    (160, 30, 10),  # oxygen
    (50, 220, 220),  # flammable
    (120, 120, 50),  # flammable-solid
    (20, 120, 50),  # corrosive
    (50, 180, 150),  # dangerous
    (40, 40, 160),  # non-flammable-gas
    (210, 80, 30),  # organic-peroxide
    (120, 120, 180),  # explosive
    (255, 130, 130),  # radioactive
    (80, 170, 100),  # inhalation-hazard
    (80, 200, 20),  # spontaneously-combustible
    (120, 120, 140),  # infectious-substance
]


class YoloDetection:
    def __init__(self, weights, config, labels, input_size=(416, 416), min_confidence=0.7, nms_threshold=0.3,
                 segmentation_enabled=True):
        np.random.seed(42)

        self._net = cv2.dnn.readNetFromDarknet(config, weights)
        self._labels = open(labels).read().strip().split("\n")

        self._colors = color_list[:len(self._labels)]

        self._layer_names = self._net.getLayerNames()
        self._layer_names = [self._layer_names[i[0] - 1]
                             for i in self._net.getUnconnectedOutLayers()]
        self.input_size = input_size
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold

        self.segmentation_enabled = segmentation_enabled

        self._detection_time = 0

    def detection_time(self):
        return self._detection_time

    def detect(self, image):
        start_time = time.time()
        blob = cv2.dnn.blobFromImage(
            image,
            1 / 255.0,
            self.input_size,
            swapRB=True,
            crop=False
        )
        self._net.setInput(blob)
        H, W = image.shape[:2]

        boxes = []
        for output in self._net.forward(self._layer_names):
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.min_confidence:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, x + int(width), y + int(height), confidence, class_id])

        if not boxes:
            return []

        class_ids = []
        positions = []
        confidences = []
        for box in boxes:
            p = box[:4]
            p[2] -= p[0]
            p[3] -= p[1]
            positions.append(p)
            confidences.append(float(box[4]))
            class_ids.append(box[5])

        if self.nms_threshold > 0:
            boxes = non_max_suppression(boxes, self.nms_threshold)

        objects = []
        for box in boxes:
            x, y, x2, y2 = [int(i) for i in box[:4]]
            w = x2 - x
            h = y2 - y
            x /= W
            w /= W
            h /= H
            y /= H
            confidence = box[4]
            class_id = int(box[5])
            if self.segmentation_enabled:
                segmentation = Segmentation(image, [x, y, w, h])
                points = segmentation.find_object(padding=0.2)
            else:
                points = []
            objects.append(Object(
                x, y, w, h,
                confidence=confidence,
                name=self._labels[class_id],
                color=[int(c) for c in self._colors[class_id]],
                points=points,
            ))

        self._detection_time = time.time() - start_time
        return objects
