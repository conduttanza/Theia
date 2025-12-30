# Helios -RasPi – MediaPipe Hand Gesture Recognition in Python

**Helios** is a Python computer vision project that uses **Google MediaPipe** to
perform **real-time hand tracking and hand gesture recognition** via a webcam.

The project is designed for **human–computer interaction**, gesture-based input,
and experimentation with **MediaPipe Hands** and **OpenCV**.

---

## Features

- Real-time **hand detection and tracking**
- **Hand gesture recognition** using MediaPipe
- Webcam input via **OpenCV**
- Lightweight and beginner-friendly
- Suitable for HCI, automation, and computer vision experiments

---

## How It Works

Helios uses **MediaPipe Hands**, a machine-learning based framework developed by
Google, to detect 21 hand landmarks per hand in real time.

These landmarks are processed to:
- track finger positions
- recognize hand poses
- detect simple gestures

---

## Technologies Used

- **Python**
- **MediaPipe**
- **OpenCV**
- Computer Vision
- Real-time video processing

---

## Installation
Activate a virtual environment then clone the repository:

```bash
git clone https://github.com/conduttanza/Helios.git
cd Helios
```
Install required packages:

```bash
pip install -r requirements.txt
```
Then start the program:

```bash
python Window_Controller.py
```
