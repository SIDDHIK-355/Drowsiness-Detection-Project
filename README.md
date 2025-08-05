# Drowsiness Detection System

## Project Overview
This project implements a real-time drowsiness detection system that can identify whether a person is awake or drowsy in vehicles. The system uses deep learning to detect multiple people in images/videos and alerts when drowsiness is detected.

## Features
- ✅ Detects multiple people in a single image/video
- ✅ Classifies each person as awake or drowsy
- ✅ Marks drowsy individuals with red bounding boxes
- ✅ Marks awake individuals with green bounding boxes
- ✅ Predicts age of drowsy individuals
- ✅ Pop-up alerts showing number of drowsy people and their ages
- ✅ GUI interface with image, video, and webcam support
- ✅ Real-time detection capability

## Model Performance
- **Accuracy**: 75.3% (exceeds minimum 70% requirement)
- **Training Dataset**: 4002 images (2001 awake, 2001 drowsy)
- **Validation Split**: 80-20
- **Model Architecture**: MobileNetV2 with transfer learning

### Performance Metrics
- Confusion Matrix and training graphs available in `/results` folder
- Precision: 0.76
- Recall: 0.74
- F1-Score: 0.75

