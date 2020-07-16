# DeepHAZMAT [Paper](https://arxiv.org/abs/2007.06392)
Hazardous Materials Sign Detection and Segmentation with Restricted Computational Resources

One of the most challenging and non-trivial tasks in robotics-based rescue operations is Hazardous Materials or HAZMATs sign detection within the operation field, in order to prevent other unexpected disasters. Each Hazmat sign has a specific meaning that the rescue robot should detect and interpret it to take a safe action, accordingly. 
Accurate Hazmat detection and real-time processing are the two most important factors in such robotics applications. Furthermore, we also have to cope with some secondary challengers such as image distortion problems and restricted CPU and computational resources which are embedded in a rescue robot. We propose a CNN-Based pipeline called DeepHAZMAT for detecting and segmenting Hazmats in four steps: 
- Optimising the number of input images that are fed into the CNN network
- Using the YOLOv3-tiny structure to collect the required visual information from the hazardous areas
- Hazmat sign segmentation and separation from the background using GrabCut technique
- Post-processing the result with morphological operators and convex hall algorithm. 

In spite of the utilisation of a very limited memory and CPU resources, the experimental results show the proposed method has successfully maintained a better performance in terms of detection-speed and detection-accuracy, compared with the state-of-the-art methods.

## Citation

```
@misc{2007.06392,
 Author = {Amir Sharifi and Ahmadreza Zibaei and Mahdi Rezaei},
 Title = {DeepHAZMAT: Hazardous Materials Sign Detection and Segmentation with Restricted Computational Resources},
 Year = {2020},
 Eprint = {arXiv:2007.06392},
}
```
