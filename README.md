**GestureX – Hand Gesture-Controlled Application**

A real-time application that uses a webcam to recognize hand gestures and control an on-screen object. Built with Python, OpenCV and MediaPipe.

# Features

| Gesture | Action |
|---|---|
| Pinch (thumb + index) on the object | Move |
| Two-hand pinch | Scale and rotate |
| Peace ✌️ | Change color |
| Fist ✊ | Change shape (Square / Triangle / Hexagon) |
| Open palm ✋ | Pause / Resume |
| Thumbs up 👍 | Reset object |
| `q` key | Quit |

# How it works

1. OpenCV captures live video from the webcam.
2. MediaPipe detects 21 hand landmarks per hand (up to 2 hands).
3. Finger states (open/closed) are used to recognize gestures.
4. Pinch is detected from the thumb-to-index distance, normalized by hand size.
5. The object is transformed using scale, a 2D rotation matrix and translation.

# HCI concepts covered

- **Natural user interaction:** pinch, open palm and fist gestures
- **Gesture recognition:** landmark-based finger state logic
- **Visual feedback:** mode panel, pinch indicator, object outline, on-screen controls
- **Error handling:** "No hand detected" warning, pause state, webcam check
- **Graphics transformations:** scale, rotation, translation

# Requirements

- Python 3.11
- A working webcam

# Installation

```bash
git clone https://github.com/syedazainabshah456/GestureX.git
cd GestureX
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

# Run

```bash
python gesture_app.py
```

Press `q` to quit.

# Tech stack

Python, OpenCV, MediaPipe, NumPy

# Notes

- Use good lighting and keep your hand clearly visible to the camera.
- This project uses MediaPipe 0.10.21 (the `mp.solutions` API).

# Author

Syeda Zainab Shah
Computer Science, Dawood University of Engineering and Technology (DUET), Karachi
