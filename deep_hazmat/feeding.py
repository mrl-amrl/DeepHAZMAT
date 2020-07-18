from .detector import YoloDetection


class ImageFeedingOptimisation:
    def __init__(self, detector: YoloDetection, k):
        self.q = 2 ** k
        self.p = self.q
        self.n = 0
        self.detector = detector
        self.last_objects = []

    def update(self, image):
        self.n += 1
        if self.n < self.p:
            return self.last_objects

        self.n = 0
        self.last_objects = self.detector.detect(image)
        if len(self.last_objects) > 0:
            if self.p > 1:
                self.p //= 2
        else:
            if self.p < self.q:
                self.p *= 2
        return self.last_objects
