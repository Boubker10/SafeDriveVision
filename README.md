# SafeDriveVision

## Project Description
SafeDriveVision is a computer vision project aimed at enhancing road safety. This project leverages deep learning models to detect and alert in real-time the dangerous behaviors of drivers, such as using a phone while driving or showing signs of drowsiness. The primary goal is to help reduce road accidents by alerting drivers to their potentially hazardous actions.

## Features
- **Phone Use Detection**: Utilizes the YOLOv5 model to identify drivers using their phones during driving.
- **Drowsiness Detection**: Incorporates a custom detector (sharp detector) to monitor signs of driver fatigue.
- **Real-Time Alerts**: Implements an alert system that warns the driver when a risky behavior is detected.


## Requierments

- PyTorch >= 0.4.1 (PyTorch v1.1.0 is tested successfully.)

- Python >= 3.6 (Numpy, Scipy, Matplotlib)

- Dlib 
- OpenCV (Python version, for image IO operations.)
- Cython (For accelerating depth and PNCC render.)




## 🧩 Installation and Configuration

1. **Clone the repository:** :
```
git clone https://github.com/Boubker10/SafeDriveVision
```
3. **Install the necessary dependencies:** :
 ```
 pip install -r requirements.txt
```
4. **Download the YOLOv5 checkpoint file (`yolov5m.pt`)**:

    Before running the project, make sure to download the YOLOv5 model checkpoint file `yolov5m.pt` from the official repository. You can download it [here](https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.pt).

## 🤖 How to Use
1. **2d_sparse**
```
python SafeDriveVision.py --onnx
```

https://github.com/Boubker10/SafeDriveVision/assets/116897761/e8325afc-63f3-4010-851d-f391d25b86ac



2. **3d Face Reconstruction**
```
python SafeDriveVision.py --onnx --opt 3d
```

https://github.com/Boubker10/SafeDriveVision/assets/116897761/63037957-e444-4b53-aa55-416eb4d1c29b


3. **2d V0**
```
python SafeDriveVisionV0.py

```

https://github.com/Boubker10/SafeDriveVision/assets/116897761/da0ef059-df9f-4ceb-9683-73e7fd228400


4. **Google Colab**
   
If you want to use the notebook version on Google Colab, download the file SafeDriveVision.ipynb and add it to your workspace 
   

# FaceBoxesNet Architecture

## Description
FaceBoxesNet is a neural network architecture designed for object detection. Below is the detailed description of its layers:

**Phase**: Can be either 'train' or 'test'.

**Conv1**: CRelu with 3 input channels, 24 output channels, kernel size 7x7, stride 4, padding 3.

**Pooling**: Max pooling with kernel size 3x3, stride 2, padding 1.

**Conv2**: CRelu with 48 input channels, 64 output channels, kernel size 5x5, stride 2, padding 2.

**Pooling**: Max pooling with kernel size 3x3, stride 2, padding 1.

**Inception**: Three consecutive Inception blocks.

**Detection Source 1**: Output after the Inception blocks.

**Conv3_1**: BasicConv2d with 128 input channels, 128 output channels, kernel size 1x1.

**Conv3_2**: BasicConv2d with 128 input channels, 256 output channels, kernel size 3x3, stride 2, padding 1.

**Detection Source 2**: Output after Conv3_2.

**Conv4_1**: BasicConv2d with 256 input channels, 128 output channels, kernel size 1x1.

**Conv4_2**: BasicConv2d with 128 input channels, 256 output channels, kernel size 3x3, stride 2, padding 1.

**Detection Source 3**: Output after Conv4_2.

**Multibox Layers**: Localization and classification layers applied to the detection sources.

## Inception Module

**Branch1x1**: BasicConv2d with kernel size 1x1.

**Branch1x1_2**: BasicConv2d with kernel size 1x1 after pooling.

**Branch3x3_reduce**: BasicConv2d with kernel size 1x1.

**Branch3x3**: BasicConv2d with kernel size 3x3.

**Branch3x3_reduce_2**: BasicConv2d with kernel size 1x1.

**Branch3x3_2**: BasicConv2d with kernel size 3x3.

**Branch3x3_3**: BasicConv2d with kernel size 3x3.

## BasicConv2d Module
2D Convolution followed by BatchNorm and ReLU.

## CRelu Module
2D Convolution followed by BatchNorm and concatenation of the output with its negative, then ReLU.

## Project Resource
The project resource can be found at [3DDFA](https://github.com/cleardusk/3DDFA).


![neuro](https://github.com/Boubker10/SafeDriveVision/assets/116897761/7044efec-f7ad-4749-8dec-be73e0ef172b)


## Contributions

We welcome contributions to the SafeDriveVision project! If you have suggestions for improvements or notice any issues, here's how you can contribute:

1. **Fork the Repository**: Start by forking the repository. This creates a copy of the project in your own GitHub account.

2. **Clone Your Fork**: Clone the forked repository to your local machine. This allows you to work on the files locally.
   ```bash
   git clone https://github.com/Boubker10/SafeDriveVision.git
   cd SafeDriveVision
3. **Create your Feature Branch** (git checkout -b feature/AmazingFeature)

4. **Commit your Changes**
5. ```git commit -m 'Add some AmazingFeature'```
   
6. **Push to the Branch**
 ``` git push origin feature/AmazingFeature ```

7. **Open a Pull Request**


## Authors

- **Boubker**  - [Boubker10](https://github.com/Boubker10)

## Contact 
Boubker BENNANI

LinkedIn: @boubkerbennani

Email: boubkerbennani44@gmail.com

Project Link: https://github.com/Boubker10/SafeDriveVision


## Views
![Total views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https://github.com/Boubker10/SafeDriveVision)

![Visitors](https://visitor-badge.glitch.me/badge?page_id=Boubker10.SafeDriveVision)


