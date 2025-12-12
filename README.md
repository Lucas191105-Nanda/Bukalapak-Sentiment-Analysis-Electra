# IndoELECTRA Sentiment Classification

Fine-tuning **IndoELECTRA** model for Indonesian sentiment analysis using HuggingFace Transformers and PyTorch.

## Overview

This project provides a complete pipeline for training a sentiment classification model using IndoELECTRA, a pre-trained language model specifically designed for Indonesian language tasks. The implementation supports GPU acceleration and includes comprehensive evaluation metrics.

## Features

- 🚀 GPU-accelerated training with CUDA support
- 🎯 Fine-tuning on custom sentiment datasets
- 📊 Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1-Score)
- 📈 Confusion matrix and classification reports
- 💾 Model persistence for deployment
- 🔮 Built-in prediction functionality

## Requirements

### System Requirements

- Python 3.9 - 3.12
- NVIDIA GPU with CUDA 12.1 (optional, but recommended)
- 8GB RAM minimum (16GB recommended)

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

**Verify GPU availability:**

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output: `True` if GPU is available.

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/indoelectra-sentiment.git
cd indoelectra-sentiment
```

2. **Create virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── train_electra.py          # Main training script
├── requirements.txt          # Python dependencies
├── electra_model_local/      # Saved model directory (after training)
├── train.csv                 # Dataset directory (add your data here)
└── README.md                 # Project documentation
```

## Usage

### Training

Run the training script:

```bash
python train_electra.py
```

The script automatically:
- Detects and uses GPU if available
- Loads and preprocesses the dataset
- Fine-tunes the IndoELECTRA model
- Saves the trained model to `electra_model_local/`
- Generates evaluation metrics

### Prediction

Use the trained model for inference:

```python
from train_electra import predict

# Single prediction
text = "Produk ini sangat bagus dan berkualitas!"
label, confidence = predict(text)

print(f"Sentiment: {label}")
print(f"Confidence: {confidence:.4f}")
```

**Example output:**

```
Sentiment: positive
Confidence: 0.9823
```

## Model Information

### IndoELECTRA Base

- **Source:** [ChristopherA08/IndoELECTRA](https://huggingface.co/ChristopherA08/IndoELECTRA)
- **Language:** Indonesian
- **Architecture:** ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)
- **Advantages:**
  - Optimized for Indonesian language
  - Faster training compared to BERT-based models
  - Excellent performance on downstream tasks

## Evaluation Metrics

The training script generates comprehensive evaluation reports:

- **Accuracy:** Overall classification accuracy
- **Precision:** Precision per class
- **Recall:** Recall per class
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Visual representation of predictions
- **Classification Report:** Detailed per-class metrics

## Configuration

Modify hyperparameters in `train_electra.py`:

```python
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
MAX_LENGTH = 128
```

## Dataset Format

Prepare your dataset in CSV format with the following structure:

```csv
text,label
"Produk sangat memuaskan",positive
"Kualitas mengecewakan",negative
"Biasa saja",neutral
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:

```python
BATCH_SIZE = 8  # or lower
```

### Slow Training (CPU)

Verify GPU availability and ensure PyTorch CUDA version matches your system:

```bash
python -c "import torch; print(torch.version.cuda)"
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Areas for Contribution

- Additional preprocessing techniques
- Web interface (Streamlit/Gradio)
- REST API implementation (FastAPI)
- Deployment guides (Docker, cloud platforms)
- Support for multi-label classification

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{indoelectra-sentiment,
  author = {Your Name},
  title = {IndoELECTRA Sentiment Classification},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/indoelectra-sentiment}
}
```

## Acknowledgments

- [IndoELECTRA](https://huggingface.co/ChristopherA08/IndoELECTRA) by ChristopherA08
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- Indonesian NLP Community

## Contact

For questions or suggestions, please open an issue or contact [nandaputraperbawa@gmail.com](mailto:nandaputraperbawa@gmail.com).

---

**Made with ❤️ for Indonesian NLP Research**
