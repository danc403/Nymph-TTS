Nymph: A system designed to train a multimodal model (for tasks like speech-to-text, text-to-speech, or similar audio-text applications) with a focus on incorporating prosody and sentiment information.  It involves several stages: transcription/data preparation, feature extraction, tokenizer training, and model training, with a dashboard for monitoring progress.

Here's a high-level overview of the key components and how they interact:

**1. Data Preparation and Transcription (`transcriber.py` and `main.py`)**

*   **Input:** The `main.py` script takes an audio/text file or directory as input.
*   **Transcription (Audio):** If the input is audio, the `transcriber.py` script uses `whisperx` to transcribe the audio into text.
*   **Feature Extraction:**  Whether the input is audio or text, the `transcriber.py` script extracts relevant features:
    *   **Audio:**  MFCCs (Mel-Frequency Cepstral Coefficients), Mel spectrograms, and prosodic features (pitch, energy, rate) are extracted using `librosa`. It also splits audio into clips, based on WhisperX's transcription segments
    *   **Text:** Sentiment analysis is performed using a Hugging Face Transformers pipeline (`roberta-base-go_emotions` by default).
*   **JSONL Output:**  The extracted features, transcriptions (if applicable), and metadata (start/end times, speaker labels) are stored in a JSONL (JSON Lines) file.  Each line in the JSONL file represents a segment of the audio or a text sentence, along with its associated data.

**2. Tokenizer Training (`tokenizer_trainer.py` and `tokenizer_config.json`)**

*   **Input:** The `tokenizer_trainer.py` script takes the JSONL file generated in the previous step and a configuration file (`tokenizer_config.json`).
*   **Tokenizer Training:** It trains a Byte Pair Encoding (BPE) tokenizer using the `tokenizers` library from Hugging Face. The tokenizer is trained on the text content of the JSONL file.  The configuration file specifies the vocabulary size, special tokens (like `<unk>`, `<pad>`, `<bos>`, `<eos>`, emotion labels, context tokens), and other tokenizer settings.
*   **Custom Tokens:** The tokenizer is designed to handle special tokens representing emotions (e.g., `[EMO_ANGER]`), context (e.g., `[CONTEXT_PREV_SENT]`), and other relevant information.  This allows the model to be aware of these contextual aspects during training.
*   **Context Handling:**  The tokenizer trainer includes logic to create context windows around sentences. This involves extracting surrounding sentences to provide context for the current sentence being tokenized.
*   **Output:** The trained tokenizer is saved to a file named `tokenizer.json`.

**3. Model Training (`model_trainer.py`)**

*   **Input:** The `model_trainer.py` script uses the JSONL data (presumably converted to parquet format) and the trained tokenizer (`tokenizer.json`).
*   **Data Loading (Arrow Flight):** The training data (in Parquet format) is loaded using Apache Arrow Flight.  This is a high-performance data transfer protocol that allows for efficient data streaming from a server (`stream.py`).
*   **Custom Dataset (`ArrowFlightDataset`):**  A custom dataset class (`ArrowFlightDataset`) is defined to handle loading and processing the data from Arrow Flight.  This dataset loads the features (audio, text, prosody, sentiment, speaker) and prepares them for the model.
*   **Model Architecture (`CustomTransformer` and `MultimodalModel`):**
    *   `CustomTransformer`: A custom transformer model (based on the standard transformer architecture) is defined for processing the text data.
    *   `MultimodalModel`: A multimodal model is defined that combines the text features from the transformer with the audio, prosody, and sentiment features.  It uses Multimodal Latent Autoregressive (MLA) layers to fuse the different modalities.
*   **Training Loop:** The script implements a training loop that iterates over the data, calculates the loss, updates the model's parameters, and periodically saves checkpoints.
*   **Loss Function:** The code uses MSELoss (Mean Squared Error Loss) - based on the training setup this appears to be autoencoder configuration.
*   **WebSocket Communication:** The training script uses WebSockets to send training progress updates (epoch, loss, accuracy) to a dashboard (`dashboard.html`).
*   **Checkpointing:** The model and optimizer states are saved periodically to checkpoint files, allowing training to be resumed from a previous state.

**4. Data Streaming (`stream.py`)**

*   **Arrow Flight Server:** The `stream.py` script implements an Apache Arrow Flight server.  This server is responsible for serving the training data (in Parquet format) to the training script.
*   **Data Directory:** The server reads Parquet files from a specified data directory.

**5. Dashboard (`dashboard.html`)**

*   **Visualization:**  The `dashboard.html` file provides a simple web-based dashboard to visualize the training progress.
*   **Real-time Updates:** It uses JavaScript and Chart.js to display the training loss and accuracy in real-time, receiving updates via WebSockets from the training script.

**How it Works (End-to-End):**

1.  **Data Preparation:** An audio file (or text file) is provided to `main.py`.
2.  **Transcription/Feature Extraction:** `transcriber.py` transcribes the audio (if necessary), extracts features (audio, prosody, sentiment), and creates a JSONL file.
3.  **Tokenizer Training:** `tokenizer_trainer.py` trains a tokenizer on the JSONL data and saves it as `tokenizer.json`.
4.  **Data Streaming:** The `stream.py` script starts an Arrow Flight server, making the data accessible over the network.
5.  **Model Training:** `model_trainer.py` connects to the Arrow Flight server, loads the data, and trains the multimodal model.  It sends training updates to the dashboard via WebSockets.
6.  **Monitoring:** The `dashboard.html` file displays the training progress in real-time.

**Key Concepts and Technologies:**

*   **Whisperx:**  For audio transcription.
*   **Librosa:** For audio feature extraction (MFCCs, prosody).
*   **Hugging Face Transformers:** For sentiment analysis.
*   **Hugging Face Tokenizers:** For tokenizer training.
*   **Apache Arrow Flight:**  For high-performance data streaming.
*   **PyTorch:**  For model definition and training.
*   **WebSockets:** For real-time communication between the training script and the dashboard.
*   **JSONL:**  A common format for storing structured data, especially for large datasets.
*   **Parquet:**  A columnar storage format optimized for analytics.

**Possible Use Cases:**

*   **Speech-to-Text with Emotion Recognition:**  The model could be trained to not only transcribe speech but also identify the speaker's emotional state.
*   **Text-to-Speech with Expressive Voices:**  The model could be used to generate speech with specific prosodic characteristics and emotional nuances.
*   **Dialogue Systems:**  The model could be part of a dialogue system that understands and responds to users in a more natural and emotionally aware way.

In summary, this project outlines a comprehensive pipeline for building and training a multimodal model that integrates audio, text, and contextual information. The use of Arrow Flight and WebSockets suggests a focus on scalability and real-time monitoring.


