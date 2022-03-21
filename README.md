# Camera Autocalibration Methods Using Modern Approaches in Deep Learning
Project: Experiments and improvements of [2018 CVMP DeepCalib](https://drive.google.com/file/d/1pZgR3wNS6Mvb87W0ixOHmEVV6tcI8d50/view) paper.
## Table of contents

- [Short description](#short-description)
- [Requirements](#requirements)
- [Dataset generation](#dataset-generation)
- [Training CNN](#training-deepcalib)
- [Camera calibraition](#camera-calibration)
- [Notes](#notes)
  - [Different architectures](#different-architectures)
  - [Weights](#weights)
  - [Undistortion](#undistortion)
- [Citation](#citation)

## Short description
This project relies on two papers [2018 CVMP DeepCalib] and [Deep Single Image Camera Calibration with Radial Distortion]. Almost all code and ideas derived from this paper [2018 CVMP DeepCalib]. The goal of our project is to experiment and find ways to improve the existing novel algorithm. We present several neural networks architectures which work with a single image of general scenes. Our approach builds upon Inception-v3, DenseNet-
121, ResNet50 architectures: our networks **automatically estimates the intrinsic parameters of the camera** (focal length and distortion parameter) from a **general single input image**.

## Requirements
- Python 3
- Keras
- TensorFlow
- OpenCV

## Dataset generation
We used the code for the whole data generation pipeline from [2018 CVMP DeepCalib]. You can donwload data from a Google drive [link](https://drive.google.com/drive/folders/1ooaYwvNuFd-iEEcmOQHpLunJEmo7b4NM) from which you can download manually.

## Training CNN
To train CNN you need to choose which architectures you want to use (refer to the `Section 4.3` of project.)

## Camera Calibration
To infer distortion parameter and focal length of a given camera, you can use prediction folder.

## Notes

#### Different architectures
For detailed information refer to the `Section 4.3` of our project. In short, all CNNs show approximately the same perfomance. But weights from [2018 CVMP DeepCalib] show better perfomance because [2018 CVMP DeepCalib] trained on large SUN360 dataset which is not available anymore. We trained on this data [link](https://drive.google.com/drive/folders/1ooaYwvNuFd-iEEcmOQHpLunJEmo7b4NM).

#### Weights
The weights for [2018 CVMP DeepCalib] can be found [here](https://drive.google.com/file/d/1TYZn-f2z7O0hp_IZnNfZ06ExgU9ii70T/view). The weights for our CNNs can be found [here](https://drive.google.com/file/d/1bTZCLx0EEZncfDNmxXPJbsGBf9pZP-rw/view?usp=sharing).

#### Undistortion
One way to qualitatively assess the accuracy of predicted parameters is to use those to undistort images that were used to predict the parameters. Undistortion folder contains MATLAB code to undistort multiple images from .txt file. The format of the .txt file is the following: 1st column contains `path to the image`, 2nd column is `focal length`, 3rd column is `distortion parameter`. Each row corresponds to a single image. With a simple modification you can use it on a single image by giving direct path to it and predicted parameters. However, you need to change only `undist_from_txt.m` file, not the `undistSphIm.m`.

## Citation
```
@inproceedings{bogdan2018deepcalib,
  title={DeepCalib: a deep learning approach for automatic intrinsic calibration of wide field-of-view cameras},
  author={Bogdan, Oleksandr and Eckstein, Viktor and Rameau, Francois and Bazin, Jean-Charles},
  booktitle={Proceedings of the 15th ACM SIGGRAPH European Conference on Visual Media Production},
  year={2018}
}

@inproceedings{xiao2012recognizing,
  title={Recognizing scene viewpoint using panoramic place representation},
  author={Xiao, Jianxiong and Ehinger, Krista A and Oliva, Aude and Torralba, Antonio},
  booktitle={2012 IEEE Conference on Computer Vision and Pattern Recognition},
  year={2012},
}
```
