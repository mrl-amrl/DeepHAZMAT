import cv2


def read_image(path):
    return cv2.imread(path)


def write_image(path, image):
    cv2.imwrite(path, image)


class Object:
    def __init__(self, x, y, w, h, confidence, name, color, points):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = confidence
        self.name = name
        self.color = color
        self.points = points

    def confidence_string(self):
        c = int(self.confidence * 100)
        return "{:d}%".format(c)

    def get_box(self):
        return self.x, self.y, self.w, self.h

    def get_center(self):
        cx = self.x + self.w / 2
        cy = self.y + self.h / 2
        return int(cx), int(cy)

    def update_position(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __str__(self) -> str:
        return f'x={self.x} y={self.y} w={self.w} h={self.h} score={self.confidence} name={self.name}'

    def __repr__(self) -> str:
        return str(self)

    def draw(self, image, padding=0.1):
        from deep_hazmat import visualizer
        h, w = image.shape[:2]
        visualizer.draw_lines(image, self.points)
        visualizer.draw_box(
            image,
            int(self.x * w),
            int(self.y * h),
            int(self.w * w),
            int(self.h * h),
            self.color,
            '{}:{}%'.format(self.name, int(self.confidence * 100)),
            padding=padding,
        )
