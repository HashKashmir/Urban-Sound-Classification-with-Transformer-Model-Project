# Urban Sound Classification with Transformer Models

This project implements a Transformer-based deep learning model to classify environmental sound events using the UrbanSound8K dataset. By converting audio signals into log-mel spectrograms and processing them with a self-attention-based architecture, the model learns to distinguish between 10 types of urban sounds (e.g., sirens, dog barks, drilling).

The model is built in **PyTorch**, and includes an end-to-end pipeline covering preprocessing, data augmentation, training, evaluation, and inference. Extensive evaluation using **stratified 10-fold cross-validation** reveals the model's strengths, weaknesses, and generalization capacity.

## Key Features

- **Pure Transformer Model** (no convolution)
- **Stratified 10-Fold Cross-Validation** for fair model assessment
- **Log-Mel Spectrogram Extraction** and normalization
- **Data Augmentation** (Gaussian noise, time/frequency masking, time shifting)
- **Training Optimization**: AdamW, OneCycleLR, gradient clipping, dropout
- **Confusion Matrix + t-SNE** for performance and interpretability
- **Inference Script** for classifying new audio clips

## Dataset

- **UrbanSound8K**
- 8,732 labeled audio clips
- 10 urban sound classes
- Resampled to 22,050 Hz
- Trimmed/padded to 2.95 seconds

## Results

| Metric                  | Final Version |
|------------------------|----------------|
| Avg. Validation Accuracy | **~67.2%**     |
| Max Validation Accuracy | **75.3%**      |


## Augmentations Used

- **Gaussian Noise**
- **Time Shifting**
- **Time Masking** (SpecAugment-inspired)
- **Frequency Masking** (new in final version)

Each applied with 50% probability during training to improve generalization and reduce overfitting.

## Model Architecture

```text
[Input Spectrogram]
     ↓
Linear Projection
     ↓
Positional Encoding
     ↓
Transformer Encoder (Self-Attention × N)
     ↓
Mean Pooling
     ↓
Linear Classifier → [10 Urban Sound Classes]
```

## Tools & Libraries
- PyTorch

- librosa

- scikit-learn

- NumPy, Pandas

- Matplotlib, Seaborn

- Google Colab (for GPU training)
