class ImageFeedingOptimisation:
    def __init__(self, k, function):
        self.q = 2 ** k
        self.p = self.q
        self.n = 0
        self.last_objects = []
        self.function = function

    def update(self, image):
        self.n += 1
        if self.n < self.p:
            return self.last_objects

        self.n = 0
        self.last_objects = self.function(image)
        if len(self.last_objects) > 0:
            if self.p > 1:
                self.p //= 2
        else:
            if self.p < self.q:
                self.p *= 2
        return self.last_objects
