from os.path import join, dirname

from deep_hazmat.detector import YoloDetection
from deep_hazmat.feeding import ImageFeedingOptimisation


class DeepHAZMAT:
    NET_DIRECTORY = join(dirname(__file__), 'net')

    def __init__(self, k, min_confidence=0.8, nms_threshold=0.3, segmentation_enabled=True):
        self._detector = YoloDetection(
            weights=join(self.NET_DIRECTORY, 'yolo.weights'),
            config=join(self.NET_DIRECTORY, 'yolo.cfg'),
            labels=join(self.NET_DIRECTORY, 'labels.names'),
            input_size=(576, 576),
            min_confidence=min_confidence,
            nms_threshold=nms_threshold,
            segmentation_enabled=segmentation_enabled,
        )
        self.optimizer = ImageFeedingOptimisation(
            k=k,
            function=self._detector.detect,
        )

    def update(self, image):
        objects = self.optimizer.update(image)
        return objects
