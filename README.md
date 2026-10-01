# Face Emotion Detection

Real-time facial emotion detection using a webcam. A Convolutional Neural Network (CNN) trained on 48×48 grayscale face images classifies faces into **7 emotions**:

`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

## How it works

1. **Face detection**: OpenCV's Haar Cascade finds faces in each webcam frame.
2. **Preprocessing**: each face is cropped, converted to grayscale, resized to 48×48 and normalised to 0–1.
3. **Classification**: the trained CNN predicts the emotion and confidence, which is drawn on screen.

## Tech stack

| Tool | Used for |
|------|----------|
| Python 3.12 | Language |
| TensorFlow / Keras 3 | Building and training the CNN |
| OpenCV (contrib, 4.x) | Webcam capture and face detection |
| NumPy, Pandas | Data handling |
| scikit-learn | Label encoding |
| Jupyter Notebook | Training workflow |
| tqdm | Progress bars while loading images |

## Model

- 4 convolutional blocks (Conv2D → MaxPooling → Dropout) with 128, 256, 512 and 512 filters
- 2 dense layers (512, 256) with dropout, then a 7-class softmax output
- Optimizer: Adam · Loss: categorical cross-entropy
- Trained on CPU with **EarlyStopping** (patience 5); the best weights were saved automatically
- Validation accuracy: **XX%** (best epoch: 38)

## Dataset

[Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) on Kaggle (based on FER-2013): 28,821 training and 7,066 test images. It isn't included in this repo because of its size.

## Project structure

```
face_emotion_detection/
├── emotion_detector.ipynb   # data loading, model building and training
├── realtime.py              # live webcam emotion detection
├── emotion_model.keras      # trained model
├── requirements.txt
└── images/                  # dataset (not included, download from Kaggle)
    ├── train/
    └── test/
```

## Setup

```bash
git clone https://github.com/finlayfs07/face-emotion-detection.git
cd face-emotion-detection
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> TensorFlow does not support Python 3.14 yet, so use Python 3.12.

## Usage

**Run live detection:**
```bash
python realtime.py
```
Press **Q** to quit.

**Retrain the model:** download the dataset into `images/`, then open `emotion_detector.ipynb` and run all cells.

## Limitations

- "Disgust" is rarely predicted because the dataset has very few disgust images.
- Haar Cascade works best with a front-facing face and good lighting.

## Author

**Finlay Shayo**, Software Engineering student at APU, Malaysia
[GitHub](https://github.com/finlayfs07) · [LinkedIn](https://www.linkedin.com/in/finlay-shayo-76171a366)
