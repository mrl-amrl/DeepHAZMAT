import argparse
from distutils.util import strtobool
from glob import glob

import cv2
from imutils import resize

from deep_hazmat import DeepHAZMAT, visualizer


def read_images(images=None, video=None):
    assert not (images is None and video is None), "images and video is empty"

    if images:
        for image in glob(images):
            yield cv2.imread(image)
        return

    capture = cv2.VideoCapture(video)
    while True:
        ret, image = capture.read()
        if not ret:
            break
        yield image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, help="k value in network optimizer", default=0)
    parser.add_argument("-min_confidence", type=float, default=0.8)
    parser.add_argument("-nms_threshold", type=float, default=0.3)
    parser.add_argument("-segmentation_enabled", type=strtobool, default=True)
    parser.add_argument("-video", type=str)
    parser.add_argument("-images", type=str)
    args = parser.parse_args()
    is_video = bool(args.video) and not bool(args.images)
    if not is_video and args.k > 0:
        print("~ for images, k must be 0")
        exit(1)

    deep_hazmat = DeepHAZMAT(
        k=args.k,
        min_confidence=args.min_confidence,
        nms_threshold=args.nms_threshold,
        segmentation_enabled=args.segmentation_enabled,
    )

    for image in read_images(images=args.images, video=args.video):
        image = resize(image, width=640)

        if is_video:
            visualizer.put_text(
                image=image,
                text=f'p={deep_hazmat.optimizer.p} q={deep_hazmat.optimizer.q} k={args.k}',
                x=10,
                y=10,
                scale=0.4,
                color=(0, 0, 0),
            )

        for hazmat in deep_hazmat.update(image):
            hazmat.draw(image=image, padding=0.1)

        cv2.imshow('image', image)
        key = cv2.waitKey(1 if is_video else 0) & 0xFF
        if key == ord('q'):
            break


if __name__ == "__main__":
    main()
