from os.path import join

from deep_hazmat.detector import YoloDetection
from deep_hazmat.feeding import ImageFeedingOptimisation


class DeepHAZMAT:
    def __init__(self, k, net_directory, min_confidence=0.8, nms_threshold=0.3,
                 segmentation_enabled=True):
        self._detector = YoloDetection(
            weights=join(net_directory, 'yolo.weights'),
            config=join(net_directory, 'yolo.cfg'),
            labels=join(net_directory, 'labels.names'),
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
