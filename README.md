## DeepHAZMAT
> [Read the Paper](https://arxiv.org/abs/2007.06392)
> [Dataset](https://drive.google.com/drive/folders/16hhlsYa0vbjmA1-nSmkOkiaUZmorY4fp)

### Abstract

Hazardous Materials Sign Detection and Segmentation with Restricted Computational Resources

One of the most challenging and non-trivial tasks in robotics-based rescue operations is Hazardous Materials or HAZMATs sign detection within the operation field, in order to prevent other unexpected disasters. Each Hazmat sign has a specific meaning that the rescue robot should detect and interpret it to take a safe action, accordingly. 
Accurate Hazmat detection and real-time processing are the two most important factors in such robotics applications. Furthermore, we also have to cope with some secondary challengers such as image distortion problems and restricted CPU and computational resources which are embedded in a rescue robot. We propose a CNN-Based pipeline called **DeepHAZMAT** for detecting and segmenting Hazmats in four steps: 
- Optimising the number of input images that are fed into the CNN network
- Using the YOLOv3-tiny structure to collect the required visual information from the hazardous areas
- Hazmat sign segmentation and separation from the background using GrabCut technique
- Post-processing the result with morphological operators and convex hull algorithm. 


<img src="https://github.com/mrl-amrl/DeepHAZMAT/blob/master/resources/banner.png?raw=true" width="100%"/>


### Dataset

We publicly release a standard Hazmat dataset with PASCAL-VOC format as a new comprehensive dataset to be used by other researchers in the field.
We also release a simple and mini dataset with different-distance images. Click [here](https://drive.google.com/drive/folders/16hhlsYa0vbjmA1-nSmkOkiaUZmorY4fp) to see the dataset.

### Google Colab Notebook

You can try it in Google Colab environment. [Click here to see notebook](https://colab.research.google.com/drive/1FW0V4T46PWydceRUyy3boxHMu5pbHsUq?usp=sharing)

#### Installation

```
$ git clone https://github.com/mrl-amrl/DeepHAZMAT
$ cd DeepHAZMAT
$ pip install .
```

#### Examples

All of these examples must be run in beside of 'net' directory. Otherwise `-net_dir` argument will be needed.

- Sample video:
```
$ python main.py -k 5 -video resources/videos/sample-0.mp4
```

- Disabling NMS:
```
$ python main.py -k 3 -video resources/videos/sample-1.mp4 -nms_threshold 0
```

- Sample image:
```
$ python main.py -images "resources/images/*"
```

- Changing minimum confidence:
```
$ python main.py -images "resources/images/*" -min_confidence 0.5
```

- Disabling segmentation:
```
$ python main.py -images "resources/images/*" -segmentation_enabled false
```

### Tests

```
$ sudo pip install nose
$ python -m nose -v --nocapture
```

### Citation

If you find this project useful in your research, please consider citing:

```
@misc{deephazmat,
 author = {Amir Sharifi and Ahmadreza Zibaei and Mahdi Rezaei},
 title = {DeepHAZMAT: Hazardous Materials Sign Detection and Segmentation with Restricted Computational Resources},
 year = {2020},
 eprint = {2007.06392},
 archivePrefix={arXiv},
 primaryClass={cs.CV}
}
```

Made in Advanced Mobile Robotics Laboratory
