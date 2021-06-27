from glob import glob
from os.path import join, basename

import cv2
from imutils import resize

from deep_hazmat import DeepHAZMAT


def main():
    base_dir = '..'
    images = list(glob(join(base_dir, 'images', '*')))
    results_folder = join(base_dir, 'results')

    deep_hazmat = DeepHAZMAT(
        k=0,
        min_confidence=0.8,
        nms_threshold=0.3,
        segmentation_enabled=True,
    )

    for image_file in images:
        file_name = basename(image_file)
        image = cv2.imread(image_file)
        image = resize(image, width=640)

        for hazmat in deep_hazmat.update(image):
            hazmat.draw(image=image, padding=0.1)

        cv2.imwrite(join(results_folder, file_name), image)


if __name__ == "__main__":
    main()
