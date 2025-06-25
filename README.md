# Emotion_Detection
## Real-Time Emotion Detector with DeepFace

A Python-based real-time facial **emotion detection system** using `OpenCV`, `DeepFace`, and `Tkinter`. This app detects facial emotions from webcam feed and displays **motivational popups** for negative emotions like sadness, anger, or fear 

---

## Features

- Real-time emotion detection using webcam
- Emotion categories: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
- Displays a motivational message when negative emotions are detected
- Live emotion overlays with colored emotion indicators
- Friendly and calming UI elements
- Emotion smoothing using recent frame history

---

##Technologies Used

- [DeepFace](https://github.com/serengil/deepface) – facial emotion recognition
- OpenCV – webcam access and video processing
- Tkinter – GUI popups
- Python – language of choice

---

##Installation

###Prerequisites

- Python 3.7 or higher
- Webcam

### Install Dependencies

```bash
pip install opencv-python deepface numpy
