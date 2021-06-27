import unittest
from os.path import basename, join
from glob import glob

from deep_hazmat import DeepHAZMAT, read_image


class DetectorTest(unittest.TestCase):
    """
    Test methods in yolo detection
    """

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        self.deep_hazmat = DeepHAZMAT(k=0, net_directory='net')

    def test_hazmats(self):
        for item in glob("resources/images/*.jpg"):
            item_name = basename(item).replace(".jpg", "")
            print("Working on {} ...".format(item_name))
            image = read_image(item)
            results = list(self.deep_hazmat.update(image))
            self.assertEqual(len(results), 1)

            result = results[0]
            self.assertEqual(item_name, result.name)


if __name__ == '__main__':
    unittest.main()
