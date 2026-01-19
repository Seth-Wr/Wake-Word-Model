🎙️ Sheila: On-Device Wake-Word Engine
A production-ready native audio trigger built with TensorFlow Lite.

This project features a custom-engineered pipeline for real-time keyword spotting. It includes a specialized ETL process for synthesizing "dirty" training data, a CNN-based classification model, and an optimized TFLite conversion for low-latency mobile inference.

Demo
https://www.youtube.com/watch?v=aabRp43wcIk

📂 Project Structure & Workflow
The engine is built across four core stages, each documented in a dedicated notebook:

File Name	Stage	Key Technical Contribution
Collect_Recordings.ipynb	Data Ingestion	Real-time audio capture via PyAudio. Captures unique ambient noise tailored for the negative class.
Data_Augmenting.ipynb	Preprocessing	Performs 1.0s windowing and MUSAN noise injection. Includes a statistical analysis cell to validate dataset duration consistency.
training_model.ipynb	Model & Export	CNN training with Binary Crossentropy and Sigmoid activation. Handles the standard SavedModel to TFLite (Float32) conversion.
TestingAudio.ipynb	Edge Validation	Real-time testing script. Compares standard and TFLite model performance on live audio streams.
🛠️ Data Engineering Strategy
A wake-word model is only as good as its training data. This project emphasizes a Robustness-First approach:

Synthetic Noise Injection: Uses the MUSAN dataset to inject music, speech, and ambient noise into positive samples at random intervals to prevent overfitting.

SNR-Balanced Augmentation: Dynamically mixes voice and noise at varying Signal-to-Noise Ratios (0dB to 15dB) to simulate diverse environments.

Data Pruning: The Data_Augmenting pipeline automatically analyzes sample lengths and prunes clips falling outside the target 0.6s–1.0s range to maintain feature uniformity.

Tailored Negative Class: Custom recordings from the user's local environment ensure the model ignores specific household or office background noises.

🧠 Model Architecture & Edge Optimization
Architecture: Convolutional Neural Network (CNN) optimized for audio frequency patterns.

Feature Engineering: Raw audio signals are converted into Log-Mel Spectrograms (40 Mel bands) to extract spectral energy features.

Edge Deployment: The final model is exported as a FlatBuffer (.tflite), bypassing the need for cloud-based inference and ensuring user privacy.

🚦 Getting Started
Dependencies: pip install tensorflow librosa torchaudio sounddevice pyaudio soundfile

Generate Data: Run Collect_Recordings.ipynb to capture background noise and Data_Augmenting.ipynb to synthesize the training set.

Train: Use training_model.ipynb to produce your sheila_float32.tflite model.

Deploy: Run TestingAudio.ipynb to test the trigger word in real-time.
