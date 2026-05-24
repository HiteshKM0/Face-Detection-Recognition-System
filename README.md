# Face Detection & Recognition System

A complete end-to-end face detection and recognition system built with OpenCV and a custom LeNet-based Convolutional Neural Network (CNN). The system captures facial images in real time via an IP camera, preprocesses and consolidates the dataset, trains a deep learning model, and performs identity recognition.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Model Architecture](#model-architecture)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## Overview

This project implements a full face recognition pipeline across three stages:

1. **Data Collection** — Captures 100 face images per person from a live IP camera stream using Haar Cascade face detection.
2. **Data Consolidation** — Reads all collected images, resizes them to 100x100 grayscale, labels them by person name, and serializes them with `pickle`.
3. **Model Training** — Preprocesses the images (histogram equalization and normalization), encodes labels, and trains a LeNet-inspired CNN using Keras.

---

## Project Structure

```
FaceDetection/
|
|-- collect_data.py            # Step 1: Capture 100 face images per person via IP camera
|-- consolidated_data.py       # Step 2: Preprocess images and serialize to pickle files
|-- Face_Detection.ipynb       # Step 3: Train the CNN model (Google Colab / Jupyter)
|
|-- images/                    # Raw captured face images (named as <person>_<index>.jpg)
|   |-- aditya_2.jpg
|   |-- aditya_3.jpg
|   `-- ...
|
|-- data/
|   |-- images.p               # Pickled numpy array of preprocessed images
|   `-- labels.p               # Pickled numpy array of corresponding labels
|
|-- final_model.h5             # Trained Keras CNN model (saved weights + architecture)
|-- haarcascade_frontalface_default.xml   # OpenCV pre-trained Haar Cascade classifier
|
`-- .spyproject/               # Spyder IDE project configuration files
```

---

## Pipeline

```
IP Camera Stream (IP Webcam App)
        |
        v
+-------------------------+
|    collect_data.py      |  <-- Haar Cascade detects faces, saves 100 images per person
+-------------------------+
        |
        v
+-------------------------+
| consolidated_data.py    |  <-- Resize -> Grayscale -> Pickle (images.p + labels.p)
+-------------------------+
        |
        v
+------------------------------------------+
|  Face_Detection.ipynb                    |
|  ----------------------------------------|
|  1. Load pickled data                    |
|  2. Label encode  (LabelEncoder)         |
|  3. Histogram equalization               |
|  4. Normalize (divide by 255)            |
|  5. One-hot encode labels                |
|  6. Train LeNet CNN (20 epochs)          |
|  7. Save  ->  final_model.h5             |
+------------------------------------------+
```

> **Note on pipeline diagram:** You can replace the ASCII diagram above with a proper flowchart image (e.g., exported from draw.io or generated with matplotlib) and embed it here as `![Pipeline](docs/pipeline.png)`.

---

## Model Architecture

The model is a custom LeNet-inspired CNN implemented using the Keras Sequential API.

| Layer | Type | Details |
|-------|------|---------|
| 1 | Conv2D | 30 filters, 5x5 kernel, ReLU activation |
| 2 | MaxPooling2D | 2x2 pool size |
| 3 | Conv2D | 15 filters, 3x3 kernel, ReLU activation |
| 4 | MaxPooling2D | 2x2 pool size |
| 5 | Flatten | — |
| 6 | Dense (Hidden) | 50 units, ReLU activation |
| 7 | Dense (Output) | N units (one per person), Softmax activation |

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (learning rate = 0.01) |
| Loss Function | Categorical Cross-Entropy |
| Epochs | 20 |
| Validation Split | 10% |
| Input Shape | (100, 100, 1) |

> **Note on model diagram:** You can embed a visual summary of the model architecture here using `model.summary()` output or a Keras visualization tool such as `keras.utils.plot_model`. Save the image and embed it as `![Model Architecture](docs/model_architecture.png)`.

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.x | Core programming language |
| OpenCV | Face detection (Haar Cascade), image capture and processing |
| Keras / TensorFlow | CNN model building and training |
| Scikit-learn | Label encoding (LabelEncoder) |
| NumPy | Array manipulation |
| Matplotlib | Image visualization |
| Pickle | Dataset serialization |
| Jupyter Notebook | Interactive model training environment |
| Spyder IDE | Local script development |

---

## Setup and Installation

### Prerequisites

- Python 3.7 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/FaceDetection.git
cd FaceDetection
```

### 2. Install Dependencies

```bash
pip install opencv-python numpy scikit-learn tensorflow keras matplotlib
```

### 3. IP Camera Setup (for Data Collection)

This project uses the **IP Webcam** app by **Thyoni Tech** (formerly Pavel Khlebovich) to stream the phone camera over a local Wi-Fi network.

- Download: [IP Webcam on Google Play](https://play.google.com/store/apps/details?id=com.pas.webcam)
- Rated 3.9/5 with over 10 million downloads.

**Setup steps:**

1. Install IP Webcam on your Android phone.
2. Connect your phone and PC to the same Wi-Fi network.
3. Open the app and tap "Start server" at the bottom.
4. Note the IP address shown on screen (e.g., `192.168.1.102:8080`).
5. Update the URL in `collect_data.py`:

```python
url = "http://<YOUR_PHONE_IP>:8080/shot.jpg"
```

You can verify the stream is working by opening the URL in a browser on your PC. You should see a live JPEG frame from your phone camera.

---

## Usage

### Step 1 — Collect Face Data

Run this script to capture 100 face images for a new person:

```bash
python collect_data.py
```

- A live window will show the detected face and a counter.
- Once 100 images are captured, enter the person's name when prompted.
- Images are saved to the `images/` folder as `<name>_0.jpg` through `<name>_99.jpg`.
- Press `q` to quit early.

### Step 2 — Consolidate and Serialize Dataset

After collecting images for all persons, run:

```bash
python consolidated_data.py
```

This will:
- Read all images from `images/`
- Resize each to 100x100 pixels and convert to grayscale
- Extract person labels from filenames (the part before the underscore)
- Serialize to `data/images.p` and `data/labels.p`

### Step 3 — Train the Model

Open and run `Face_Detection.ipynb` locally or in Google Colab:

```bash
jupyter notebook Face_Detection.ipynb
```

If using Google Colab, upload `images.p` and `labels.p` to the `/content/` directory before running.

The notebook will:
- Load and preprocess the data
- Train the LeNet CNN for 20 epochs
- Save the trained model as `final_model.h5`

### Step 4 — Load and Use the Model

```python
from keras.models import load_model
import numpy as np
import cv2

model = load_model('final_model.h5')

# Preprocess a new face image
img = cv2.imread('test_face.jpg')
img = cv2.resize(img, (100, 100))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.equalizeHist(img)
img = img.reshape(1, 100, 100, 1) / 255.0

prediction = model.predict(img)
predicted_class = np.argmax(prediction)
print(f"Predicted Person Index: {predicted_class}")
```

---

## Dataset

| Property | Details |
|----------|---------|
| Images per person | 100 (captured via IP camera) |
| Image format | JPEG |
| Stored format | Grayscale, resized to 100x100 pixels |
| Serialization | Python pickle (.p files) |
| Naming convention | person_name_index.jpg |
| Current dataset | 1 person (aditya), 34 collected images |

> **Note:** The `images/` folder currently contains images for a single person. For better model generalization and accuracy, collect data for multiple individuals (recommended: 3 or more persons, 100 images each).

> **Note on sample images:** You may embed a few representative face crops from the dataset here to give readers a visual sense of the data quality and variety. Example: `![Sample Faces](docs/sample_faces.png)`.

---

## Results

- The model is trained with a 90/10 train-validation split over 20 epochs.
- With sufficient data (100 images per person across multiple individuals), the LeNet architecture converges reliably within the training duration.
- The trained model is saved to `final_model.h5` and can be loaded directly for inference.

> **Note on training plots:** You can save the training accuracy and loss curves from the notebook and embed them here as `![Training History](docs/training_history.png)`. This helps readers quickly assess model performance.
