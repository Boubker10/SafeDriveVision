## Requierments

- PyTorch >= 0.4.1 (PyTorch v1.1.0 is tested successfully on macOS and Linux.)

- Python >= 3.6 (Numpy, Scipy, Matplotlib

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
   











https://github.com/Boubker10/SafeDriveVision/assets/116897761/da0ef059-df9f-4ceb-9683-73e7fd228400
