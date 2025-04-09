# Indian Sign Language Translator

A deep learning-based system for translating speech/text to Indian Sign Language (ISL) using computer vision and natural language processing.

## Features

- Speech-to-Sign Language translation
- Text-to-Sign Language translation
- Real-time hand gesture recognition
- Support for Indian Sign Language vocabulary
- Web interface for easy interaction

## Project Structure

```
├── data/                   # Dataset directory
│   ├── raw/               # Raw video and image data
│   └── processed/         # Processed and augmented data
├── models/                # Trained model files
├── src/                   # Source code
│   ├── data_processing/   # Data preprocessing scripts
│   ├── model/            # Model architecture and training
│   ├── utils/            # Utility functions
│   └── web/              # Web interface
├── notebooks/            # Jupyter notebooks for experimentation
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the ISL dataset and place it in the `data/raw` directory

4. Run the data preprocessing script:
```bash
python src/data_processing/preprocess.py
```

5. Train the model:
```bash
python src/model/train.py
```

6. Start the web interface:
```bash
python src/web/app.py
```

## Model Architecture

The system uses a combination of:
- CNN for hand gesture recognition
- LSTM for temporal sequence processing
- Transformer for text-to-sign sequence generation
- MediaPipe for hand landmark detection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
