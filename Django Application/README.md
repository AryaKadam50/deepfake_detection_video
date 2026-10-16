# Deepfake Detection Django Application

## Requirements

**Note:** An NVIDIA GPU is recommended/required to run GPU-accelerated inference.
- CUDA version >= 10.0 for GPU
- GPU Compute Capability > 3.0 

You can find the list of dependencies in [requirements.txt](requirements.txt). Main requirements include:

```
Python >= v3.6
Django >= v3.0
```

## Directory Structure

- `ml_app/` — Django application containing views, models, and prediction logic
- `project_settings/` — Django configuration and settings
- `static/` — Static assets (CSS, JS, and weights for client-side face detection)
- `templates/` — HTML template files

> **Note:** Before running the project, ensure directories named `models`, `uploaded_images`, and `uploaded_videos` exist in the project root with write permissions.

---

## Running with Docker

#### Step 1: Install Docker Desktop and start the Docker daemon

#### Step 2: Run the Deepfake Detection container
```bash
docker run --rm --gpus all -v static_volume:/home/app/staticfiles/ -v media_volume:/app/uploaded_videos/ --name=deepfakeapplication deepfake-detection-20framemodel
```

#### Step 3: Run the Nginx reverse proxy server container
```bash
docker run -p 80:80 --volumes-from deepfakeapplication -v static_volume:/home/app/staticfiles/ -v media_volume:/app/uploaded_videos/ deepfake-nginx-proxyserver
```

#### Step 4: Open application in your browser
Access the application at [http://localhost:80](http://localhost:80).

---

## Running Locally

### Prerequisites
Download a trained model checkpoint from [Google Drive](https://drive.google.com/drive/folders/1UX8jXUXyEjhLLZ38tcgOwGsZ6XFSLDJ-?usp=sharing) or train a custom model using the steps in the `Model Creation` directory.

#### Step 1: Navigate to the Django Application directory
```bash
cd "Django Application"
```

#### Step 2: Create a virtual environment (optional)
```bash
python -m venv venv
```

#### Step 3: Activate the virtual environment
- **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```
- **Windows:**
  ```cmd
  venv\Scripts\activate
  ```

#### Step 4: Install requirements
```bash
pip install -r requirements.txt
```

#### Step 5: Place the model weights
Copy the trained model checkpoint into `Django Application/models/`.

> **Note:** The model file name must follow the expected format, for example: `model_84_acc_10_frames_final_data.pt` (where the sequence frame length follows the third underscore).

#### Step 6: Start the Django development server
```bash
python manage.py runserver
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## Demo

<p align="center">
  <img src="../github_assets/fakegif.gif" alt="Demo GIF" />
</p>
