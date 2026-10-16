# Deepfake Video Detection using Deep Learning

An end-to-end deep learning system designed to detect facial manipulation and AI-generated deepfake videos. The system combines convolutional spatial feature extraction with recurrent temporal sequence modeling to identify visual inconsistencies and inter-frame anomalies across video sequences.

---

## What This Project Is About

Deepfakes are synthetic videos where a person's facial likeness, expressions, or identity are replaced or manipulated using generative artificial intelligence. While individually generated frames can look convincing, manipulation techniques often leave behind subtle temporal artifacts across consecutive frames — such as micro-jittering, flickering edges, unnatural eye blinking rates, and temporal blurring.

This project implements a spatio-temporal deep learning pipeline capable of analyzing video sequences and determining whether a video is **REAL** or **FAKE**, along with a prediction confidence score.

The repository includes:
1. **A Trained Deep Learning Architecture** combining a pretrained ResNeXt-50 Convolutional Neural Network (CNN) with a Long Short-Term Memory (LSTM) network.
2. **A Full-Stack Web Application (Django)** where users can upload any video, choose sequence lengths, view extracted face crops, and see real-time detection results with bounding-box tracking.
3. **Model Creation & Training Pipelines** with Jupyter notebooks covering preprocessing, face cropping, model training, and standalone inference.

---

## How It Works

The detection pipeline analyzes videos in four sequential stages:

```
[Input Video]
      │
      ▼
[1. Frame Extraction & Face Detection]  ──> Face recognition & alignment (112×112)
      │
      ▼
[2. Spatial Feature Extraction]         ──> ResNeXt-50 (32x4d) extracts 2048-dim vectors
      │
      ▼
[3. Temporal Sequence Modeling]         ──> LSTM captures frame-to-frame inconsistencies
      │
      ▼
[4. Binary Classification & Output]     ──> Output: REAL / FAKE + Confidence %
```

### 1. Frame Extraction and Face Detection
- The video is decoded into individual frames sampled at uniform intervals.
- A face detector identifies the location of faces in each frame.
- The detected facial bounding box is cropped, resized to $112 \times 112$ pixels, and normalized using standard ImageNet parameters.

### 2. Spatial Feature Extraction (ResNeXt-50)
- Each cropped face is passed through a pretrained **ResNeXt-50 (32x4d)** convolutional neural network.
- Grouped convolutions extract fine-grained facial features, micro-textures, and boundary artifacts.
- The network outputs a 2048-dimensional feature vector for each frame.

### 3. Temporal Sequence Modeling (LSTM)
- Consecutive feature vectors across time are assembled into a sequence of length $T$ (e.g., 10, 20, 40, 60, 80, or 100 frames).
- An **LSTM (Long Short-Term Memory)** layer processes the sequence chronologically, learning temporal dynamics and identifying inter-frame anomalies that are characteristic of deepfake synthesis.

### 4. Classification & Prediction
- The final state of the sequence is passed through a dropout layer ($p = 0.4$) and a linear classification head.
- Softmax outputs the probability distribution across two classes: **REAL** or **FAKE**, accompanied by a confidence percentage.

---

## System Architecture

<p align="center">
  <img src="github_assets/System%20Architecture.png" alt="System Architecture Diagram" width="85%" />
</p>

---

## Detection Performance & Results

The architecture was evaluated across 6,000 video samples with various sequence frame lengths ($T$). As the sequence length increases, the model captures longer temporal context, resulting in higher classification accuracy:

| Model Checkpoint | Videos Evaluated | Sequence Length | Accuracy |
|---|---|---|---|
| `model_84_acc_10_frames_final_data.pt` | 6,000 | 10 frames | **84.21%** |
| `model_87_acc_20_frames_final_data.pt` | 6,000 | 20 frames | **87.79%** |
| `model_89_acc_40_frames_final_data.pt` | 6,000 | 40 frames | **89.35%** |
| `model_90_acc_60_frames_final_data.pt` | 6,000 | 60 frames | **90.59%** |
| `model_91_acc_80_frames_final_data.pt` | 6,000 | 80 frames | **91.50%** |
| `model_93_acc_100_frames_final_data.pt`| 6,000 | 100 frames | **93.59%** |

Analyzing individual frames (10 frames) achieves ~84.2% accuracy, whereas analyzing longer sequences (100 frames) improves detection accuracy to **93.59%**, demonstrating that temporal continuity plays a crucial role in distinguishing deepfakes from genuine video.

---

## Directory Structure

```
deepfake_detection_video/
├── README.md                           # Main project documentation
├── .gitignore                          # Git ignore rules for build and cache files
├── github_assets/
│   └── System Architecture.png         # Pipeline architecture diagram
│
├── Django Application/                 # Web interface for inference
│   ├── manage.py                       # Django management script
│   ├── requirements.txt                # Python dependencies
│   ├── Dockerfile                      # Application Docker container definition
│   ├── project_settings/               # Project settings, URL routing, and WSGI configuration
│   ├── ml_app/                         # Machine learning application views and controllers
│   │   ├── views.py                    # Inference logic, video processing, and model loading
│   │   ├── forms.py                    # Video file upload form and sequence slider
│   │   ├── urls.py                     # Web route definitions
│   │   └── templates/                  # Frontend HTML templates (index, predict, 404)
│   ├── nginx/                          # Nginx reverse proxy Dockerfile and configuration
│   ├── static/                         # CSS, JavaScript, and Face-API client-side model files
│   ├── templates/                      # Base page layouts and navigation bars
│   ├── models/                         # Storage directory for PyTorch .pt model checkpoints
│   ├── uploaded_videos/                # Upload buffer for test videos
│   └── uploaded_images/                # Buffer for split and cropped facial frames
│
└── Model Creation/                     # Model development and experimentation notebooks
    ├── preprocessing.ipynb             # Face extraction and video frame splitting notebook
    ├── Model_and_train_csv.ipynb       # ResNeXt + LSTM model architecture and training loop
    ├── Predict.ipynb                   # Standalone inference script for test videos
    ├── Readme.md                       # Guide for running training pipelines
    ├── labels/
    │   └── Gobal_metadata.csv          # Metadata and ground-truth labels for dataset splits
    └── Helpers/
        ├── Create_csv_from_glob.ipynb  # Helper script to create metadata CSVs from raw paths
        ├── label_json_to_csv.py        # Converts JSON annotations into structured CSV format
        ├── for_Balancing_data.ipynb    # Class balancing and sampling utilities
        ├── copy real and fake .ipynb   # Dataset organizing script
        └── Remove_audio_altered_files.ipynb # Utility to filter audio-modified videos
```

---

## Running the Web Application

### Option 1: Running Locally

1. **Navigate to the Django Application directory:**
   ```bash
   cd "Django Application"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add trained model weights:**
   - Place your `.pt` model file inside `Django Application/models/`.
   - Name format: `model_<acc>_acc_<frames>_frames_final_data.pt` (e.g., `model_87_acc_20_frames_final_data.pt`).

5. **Start the development server:**
   ```bash
   python manage.py runserver
   ```
   Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

### Option 2: Running with Docker

1. **Build and start the application container:**
   ```bash
   cd "Django Application"
   docker build -t deepfake-app .
   docker run --rm --gpus all \
     -v static_volume:/home/app/staticfiles/ \
     -v media_volume:/app/uploaded_videos/ \
     --name=deepfakeapplication deepfake-app
   ```

2. **Run the Nginx reverse proxy:**
   ```bash
   cd nginx
   docker build -t deepfake-nginx .
   docker run -p 80:80 \
     --volumes-from deepfakeapplication \
     -v static_volume:/home/app/staticfiles/ \
     -v media_volume:/app/uploaded_videos/ \
     deepfake-nginx
   ```

3. Access the web interface at [http://localhost](http://localhost).

---

## Model Training & Experimentation

The `Model Creation/` folder contains Jupyter notebooks covering the full machine learning lifecycle:

1. **Preprocessing (`preprocessing.ipynb`):**
   - Decodes raw videos from deepfake benchmark datasets (e.g., FaceForensics++, Celeb-DF, DFDC).
   - Extracts frames at regular intervals and crops the facial bounding boxes.
   - Saves the cropped face videos for efficient training.

2. **Training & Validation (`Model_and_train_csv.ipynb`):**
   - Loads cropped sequences alongside labels from `labels/Gobal_metadata.csv`.
   - Configures the pretrained ResNeXt-50 feature extractor and LSTM layer.
   - Trains using Cross-Entropy Loss with the Adam optimizer.
   - Evaluates test accuracy and loss convergence across epochs.

3. **Inference (`Predict.ipynb`):**
   - Loads a trained checkpoint and runs predictions on standalone video files without requiring the web server.

---

## Technologies Used

- **Deep Learning:** PyTorch, Torchvision (ResNeXt-50), LSTM
- **Computer Vision:** OpenCV, `face_recognition`, PIL
- **Web Backend:** Django, Gunicorn, Python
- **Frontend & Visualization:** HTML5, CSS3, Bootstrap, Face-API.js
- **Deployment:** Docker, Nginx
- **Hardware Acceleration:** NVIDIA CUDA (with CPU support fallback)
