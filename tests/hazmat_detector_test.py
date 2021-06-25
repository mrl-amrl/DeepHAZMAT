import unittest
from os.path import basename, join
from glob import glob

from deep_hazmat import YoloDetection, read_image


class DetectorTest(unittest.TestCase):
    """
    Test methods in yolo detection
    """

    net_directory = join('deep_hazmat', 'net')

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        self.detector = YoloDetection(
            weights=join(self.net_directory, 'yolo.weights'),
            config=join(self.net_directory, 'yolo.cfg'),
            labels=join(self.net_directory, 'labels.names'),
            input_size=(576, 576),
            min_confidence=0.75,
            nms_threshold=0.3,
        )

    def test_hazmats(self):
        for item in glob("resources/images/*.jpg"):
            item_name = basename(item).replace(".jpg", "")
            print("Working on {} ...".format(item_name))
            image = read_image(item)
            results = self.detector.detect(image)
            self.assertEqual(len(results), 1)

            result = results[0]
            self.assertEqual(item_name, result.name)


if __name__ == '__main__':
    unittest.main()
