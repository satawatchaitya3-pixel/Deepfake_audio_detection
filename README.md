# Deepfake Audio Detection Project

## Overview
Deepfake Voice Detection System is a machine learning–based application designed to identify whether an audio sample is genuine or artificially generated (deepfake). With the rapid advancement of AI-driven voice synthesis and cloning technologies, detecting synthetic audio has become a critical challenge in cybersecurity and digital forensics.

This project addresses the problem by analyzing audio signals using Mel-Frequency Cepstral Coefficients (MFCC) and classifying them using a Support Vector Machine (SVM) model. The system processes audio input, extracts meaningful features, and predicts whether the voice is real or fake.

A Flask-based web interface is also integrated, allowing users to upload audio files and receive instant detection results. The project demonstrates a practical implementation of machine learning techniques for combating deepfake audio threats.

Overall, this system provides a simple, efficient, and scalable solution for enhancing audio authenticity and security in digital communication.

## Citation
Satawat, C. (2026). Deepfake Voice Detection System using Machine Learning.
GitHub Repository: https://github.com/satawatchaitya3-pixel/deepfake-voice-detection

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [How to Use](#how-to-use)
   - [Training the Model](#training-the-model)
   - [Analyzing Audio](#analyzing-audio)
4. [License](#license)

## Installation
To initialize the project, follow these steps:

1. Clone the repository to your local machine:
   ```
   git clone https://github.com/your-username/deepfake-audio-detection.git
   cd deepfake-audio-detection
   ```

2. Set up a virtual environment (optional but recommended):
   ```
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies

## How to Use Training the Model
To train the SVM model with the provided data, follow these steps:

1. Prepare the dataset:
   Place genuine audio files in the `real_audio` directory and deepfake audio files in the `deepfake_audio` directory.

2. Run the training script:
   ```
   python main.py
   ```
   After sucessfully running the main script, it will initially ask you to provide the path of the voice to analyze, provide it with the path and the
3. Run the web app by:
   ```
   python app.py
   ```

   The training script will extract MFCC features from the audio files, split the data into training and testing sets, scale the features, train the SVM model, and save the trained model and scaler for future use.

### Analyzing Audio
To classify an audio file as genuine or deepfake, follow these steps:

1. Ensure the trained model and scaler are available (already saved during training).

2. Run the analysis script:
   ```
   python analyze_audio.py path/to/your/audio/file.wav
   ```

   Replace `path/to/your/audio/file.wav` with the path to the audio file you want to analyze. The script will extract MFCC features from the audio, scale the features using the saved scaler, pass the features to the trained SVM model, and display the classification result.


## Contribution & License
- For contributing, fork this project and compare and submit a pull request with proper description to your changed/added features
- OpenSource MIT License, for more information read the [License](./LICENSE).
