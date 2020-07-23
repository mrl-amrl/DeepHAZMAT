from os.path import join, dirname

from .detector import YoloDetection
from .feeding import ImageFeedingOptimisation


class DeepHAZMAT:
    NET_DIRECTORY = join(dirname(__file__), 'net')

    def __init__(self, k, min_confidence=0.8):
        self._detector = YoloDetection(
            weights=join(self.NET_DIRECTORY, 'yolo.weights'),
            config=join(self.NET_DIRECTORY, 'yolo.cfg'),
            labels=join(self.NET_DIRECTORY, 'labels.names'),
            input_size=(576, 576),
            min_confidence=min_confidence,
            nms_threshold=0.3,
        )
        self.optimizer = ImageFeedingOptimisation(
            detector=self._detector,
            k=k,
        )

    def update(self, image):
        h, w = image.shape[:2]
        for hazmat_object in self.optimizer.update(image):
            hazmat_object.x /= w
            hazmat_object.w /= w
            hazmat_object.y /= h
            hazmat_object.h /= h
            yield hazmat_object
