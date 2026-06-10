# AI-Powered Driver Drowsiness Detection System

Real-time driver fatigue monitoring using MediaPipe Face Mesh, OpenCV, and Eye Aspect Ratio (EAR) analysis.

## Features

- Real-time face mesh tracking (468 facial landmarks)
- Blink detection
- Eye Aspect Ratio (EAR) monitoring
- Driver drowsiness detection
- Audio alert system
- Visual warning system
- Live webcam processing

## Technologies Used

- Python
- OpenCV
- MediaPipe
- Computer Vision
- Facial Landmark Detection

## How It Works

1. Captures video from webcam
2. Detects facial landmarks
3. Calculates Eye Aspect Ratio (EAR)
4. Detects prolonged eye closure
5. Triggers audio and visual alerts

## Future Improvements

- Raspberry Pi integration
- In-car deployment
- Mobile app support
- Driver monitoring dashboard

# Face Mesh

A simple Python script that detects and overlays a 468-point light blue mesh on your face in real-time using your webcam.

## How to Use

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python3 face.py
   ```

*Note: Press the `Esc` key while the camera window is in focus to securely close the application.*
