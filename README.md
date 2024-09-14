# Image Processing and Detection Pipeline

## Overview

This project implements an image augmentation pipeline and face detection system. It can apply transformations like brightness, contrast, and saturation adjustments to images. It also provides face detection and recognition capabilities using pre-trained models.

***

## Features

- **Image Augmentation**: 
  - Adjusts brightness, contrast, and saturation.
  - Saves augmented images to the specified directory.
  
- **Face Detection and Recognition**:
  - Detects faces in images.
  - Recognizes faces using known encodings.
  - Annotates images with bounding boxes and recognized names.

***

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/face-recognition.git
   cd repo-name

1. **Install the Dependencies**: To install the necessary Python packages, use:
2. ```bash
   pip install -r requirements.txt

3. **Install RetinaFace** (optional, if using face detection):
  ```bash
    pip install retinaface

## Usage

### 1. Image Augmentation

To run the image augmentation pipeline:

```bash
python src/Processing_and_Augmentation_Pipeline.py --image_path path/to/image --output_folder path/to/output

