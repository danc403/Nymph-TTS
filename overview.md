# Nymph: Multimodal Language Model Training Pipeline

## Overview

Nymph is a comprehensive pipeline designed to train a multimodal language model capable of processing and integrating information from text, audio, prosody, sentiment, and speaker identity. Our goal is to build a model that can leverage these diverse data streams to perform tasks like speech-to-text with emotional understanding, expressive text-to-speech, and more natural dialogue systems. The pipeline emphasizes efficient data handling, flexible model architecture, and real-time training monitoring.

## Key Features

*   **Multimodal Data Integration:** Processes and combines text, audio, prosody (pitch, energy, rate), sentiment, and speaker identity into a unified representation.
*   **Customizable Model Architecture:**  Employs a custom Transformer architecture (including Multimodal Latent Attention and Sliding Window Attention) designed for large context windows and tailored to multimodal data.
*   **Efficient Data Streaming:** Leverages Apache Arrow Flight for high-performance data transfer between the data serving and training servers.  Supports concurrent loading from multiple Parquet files.
*   **Flexible Tokenization:** Utilizes a Byte Pair Encoding (BPE) tokenizer, trained with custom tokens for sentiment, prosody, and context, enabling fine-grained control and representation.
*   **Real-time Training Monitoring:** Provides a web-based dashboard that displays training progress (loss, accuracy) in real-time via WebSocket communication.
*   **Checkpointing:** Regularly saves model and optimizer states to enable resuming training from previous checkpoints.
*   **Modular Design:**  Each stage of the pipeline (data preparation, tokenization, model training, data streaming, monitoring) is implemented as a separate module, promoting maintainability and extensibility.
*   **Large Context Window Support:** Designed to handle large context windows (256k tokens) using efficient attention mechanisms.

## System Architecture

The Nymph training pipeline consists of the following key components:

*   **Data Preparation (`transcriber.py`, `main.py`):**
    *   Accepts audio, text, or JSONL files as input.
    *   Transcribes audio using WhisperX (if necessary).
    *   Extracts audio features (MFCCs, mel-spectrograms, prosody) using Librosa.
    *   Performs sentiment analysis using a Hugging Face Transformers pipeline.
    *   Generates a JSONL file containing extracted features, transcriptions, and metadata.
*   **Tokenizer Training (`tokenizer_trainer.py`, `tokenizer_config.json`):**
    *   Trains a BPE tokenizer based on the JSONL data and a configuration file (`tokenizer_config.json`).
    *   Incorporates custom tokens for emotions, prosody, context, and punctuation.
    *   Saves the trained tokenizer to `tokenizer.json`.
*   **Data Streaming (`stream.py`):**
    *   Implements an Apache Arrow Flight server to efficiently stream training data (in Parquet format) to the training server.
    *   Handles concurrent access to multiple Parquet files.
*   **Model Training (`model_trainer.py`):**
    *   Loads training data from the Arrow Flight server using a custom dataset class (`ArrowFlightDataset`).
    *   Utilizes a custom Transformer-based model architecture (`CustomTransformer`, `MultimodalModel`).
    *   Trains the model using PyTorch, optimizing a suitable loss function (currently MSELoss, indicating an autoencoder configuration).
    *   Sends training progress updates to the dashboard via WebSockets.
    *   Saves model checkpoints periodically.
*   **Dashboard (`dashboard.html`):**
    *   Provides a web interface for visualizing training progress in real-time.
    *   Displays training loss and accuracy using Chart.js.

## Hardware Setup

The pipeline is designed to be distributed across two machines:

*   **Training Server (Threadripper):**
    *   CPU: 12-core Threadripper
    *   RAM: 128GB
    *   GPUs: Two 24GB P40 GPUs, Six 8GB CMP 104-100 GPUs
    *   Purpose: Model training and optimization.
*   **Data Serving Server (Ryzen):**
    *   CPU: 6-core Ryzen
    *   RAM: 100GB
    *   GPU: 8GB 4060 GPU
    *   Storage: NVMe drive for data storage
    *   Purpose: Serving pre-tokenized Parquet data via Arrow Flight.

## Data Flow

1.  Raw data (audio, text) is processed by the data preparation scripts to generate JSONL files containing extracted features.
2.  The JSONL files are converted to Parquet format for efficient storage and retrieval.
3.  The Ryzen machine serves the Parquet data via the Arrow Flight server.
4.  The Threadripper machine connects to the Arrow Flight server, loads the data, and trains the multimodal model.
5.  Training progress is monitored via the web-based dashboard.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd nymph
    ```
2.  **Install dependencies:** (Provide instructions for installing necessary Python packages and other dependencies)
3.  **Configure the data serving server (Ryzen):**
    *   Set the `data_dir` variable in `stream.py` to the directory containing your Parquet data files.
    *   Run the Arrow Flight server: `python stream.py`
4.  **Configure the training server (Threadripper):**
    *   Edit `model_trainer.py` to specify the correct host IP address and port of the Arrow Flight server, and the list of parquet files to use.
    *   Start the training process: `python training/model_trainer.py --host <ryzen_ip> --port 9090 --file_list dataset1.parquet dataset2.parquet ...`
5.  **Open the dashboard:**
    *   Open `dashboard.html` in your web browser.  The dashboard will connect to the training server and display training progress in real-time.

## Technologies Used

*   **Python:**  Primary programming language.
*   **PyTorch:** Deep learning framework.
*   **WhisperX:**  Audio transcription.
*   **Librosa:** Audio feature extraction.
*   **Hugging Face Transformers:** Sentiment analysis.
*   **Hugging Face Tokenizers:** Tokenization.
*   **Apache Arrow Flight:** Data streaming.
*   **WebSockets:** Real-time communication.
*   **JSONL:** Data storage format.
*   **Parquet:** Columnar storage format.
*   **Chart.js:**  Dashboard visualization.

## Future Directions

*   **Improved Model Architectures:** Explore advanced Transformer variants and attention mechanisms to improve model performance and efficiency.
*   **Advanced Loss Functions:**  Experiment with different loss functions to optimize for specific tasks and data modalities.
*   **Model Parallelism:** Implement model parallelism to leverage all available GPUs on the training server.
*   **Expanded Feature Set:** Incorporate additional features, such as visual information or external knowledge sources.
*   **Evaluation Metrics:** Implement robust evaluation metrics to assess model performance across various tasks.
*   **Deployment:** Develop a deployment strategy for serving the trained model in real-world applications.
*   **Active Learning:** Explore active learning techniques to improve data efficiency and model accuracy.

## Contributing

We welcome contributions to the Nymph project!  Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.
