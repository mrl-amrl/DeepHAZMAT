import cv2
import numpy as np


def find_rectangles(mask):
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hull = []
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        hull.append(cv2.convexHull(c, False))
    return hull


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def remove_noises(img, k=5):
    kernel = np.ones((k, k), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img


class Segmentation:
    def __init__(self, image, rect: list, gamma=1.0):
        self.image = image
        self.rect = rect
        self.gamma = gamma

    def find_object(self, padding=0.2):
        img_h, img_w = self.image.shape[:2]
        x, w = int(self.rect[0] * img_w), int(self.rect[2] * img_w)
        y, h = int(self.rect[1] * img_h), int(self.rect[3] * img_h)
        px = int(padding * w)
        py = int(padding * h)
        x -= px
        y -= py
        h += py * 2
        w += px * 2

        sx = max(0, int(x))
        sy = max(0, int(y))
        ex = min(img_w, int(x + w))
        ey = min(img_h, int(y + h))
        if sx > img_w or sy > img_h:
            return []

        image = self.image[sy:ey, sx:ex].copy()
        mask = np.zeros(image.shape[:2], np.uint8)
        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, mask, (1, 1, w - 1, h - 1), background_model, foreground_model, 10, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        roi = image * mask2[:, :, np.newaxis]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(roi, 10, 255, cv2.THRESH_BINARY)[1]
        return self.normalize_points(mask, x, y)

    @staticmethod
    def normalize_points(mask, x, y):
        contours = find_rectangles(mask)
        if not contours:
            return []
        output = []
        for point in find_rectangles(mask)[0]:
            output.append((point[0][0] + x, point[0][1] + y))
        return output
