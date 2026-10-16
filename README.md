# Deepfake Detection using Deep Learning (ResNeXt and LSTM)

## Latest Update
The [Django Application](Django%20Application) is dockerized — you can spin up a container within seconds without worrying about dependencies.

## 1. Introduction
This project aims to detect video deepfakes using deep learning techniques combining ResNeXt and LSTM. Deepfake detection is achieved through transfer learning, where a pretrained ResNeXt Convolutional Neural Network (CNN) extracts feature vectors from video frames, and an LSTM layer is trained on these sequential feature representations to classify videos as authentic or manipulated.

- **Demo Video:** [Watch on YouTube](https://www.youtube.com/watch?v=_q16aJTXVRE)
- **Step-by-Step Installation:** [YouTube Playlist](https://www.youtube.com/watch?v=quJ8Rv84oA0&list=PLNIj0dkfMA1FsD5xR4IEc8vdwr66_WExl)

## 2. Directory Structure
```
Deepfake_detection_using_deep_learning
    |
    |--- Django Application
    |--- Model Creation
    |--- Documentation
```

1. **Django Application**: Contains the web application where a user can upload a video and submit it to the model for prediction. The trained model performs inference and displays the result on the screen.
2. **Model Creation**: Contains the step-by-step process of preprocessing, feature extraction, and training the deepfake detection model.
3. **Documentation**: Contains documentation and architecture references for the project.

## 3. System Architecture
<p align="center">
  <img src="github_assets/System%20Architecture.png" alt="System Architecture" />
</p>

## 4. Demo 
### Watch the [YouTube Video](https://www.youtube.com/watch?v=_q16aJTXVRE&t=823s) for a demonstration

<p align="center">
  <img src="github_assets/fakegif.gif" alt="Demo GIF" />
</p>

## 5. Results

| Model Name | Number of Videos | Number of Frames | Accuracy |
|------------|------------------|------------------|----------|
| `model_84_acc_10_frames_final_data.pt` | 6000 | 10 | 84.21% |
| `model_87_acc_20_frames_final_data.pt` | 6000 | 20 | 87.79% |
| `model_89_acc_40_frames_final_data.pt` | 6000 | 40 | 89.35% |
| `model_90_acc_60_frames_final_data.pt` | 6000 | 60 | 90.59% |
| `model_91_acc_80_frames_final_data.pt` | 6000 | 80 | 91.50% |
| `model_93_acc_100_frames_final_data.pt`| 6000 | 100 | 93.59% |

## 6. License

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
