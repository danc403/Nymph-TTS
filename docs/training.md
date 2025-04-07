Project Goal:
We are building a large multimodal language model, that is designed to process audio, text, prosody, sentiment, and speaker information. The model is being designed to operate with a very large context window.
Key Components:
1. Model Architecture:
• We've moved away from using a pre-trained BERT model and are now using a custom-built Transformer architecture (CustomTransformer). This allows us to have complete control over the model's size and configuration.
• The model incorporates Multi-head Latent Attention (MLA) and Sliding Window Attention (SWA) to efficiently handle very large context windows (256k).
• The model integrates audio, prosody, sentiment, and speaker embeddings alongside the text input.
• The model is set up as an autoencoder, where the target is the combined input features.
2. Dataset Handling:
• The data is stored in Parquet format for efficient storage and retrieval.
• We've implemented an Apache Arrow Flight server on a separate machine (Ryzen) to stream the pre-tokenized Parquet data to the training server (Threadripper).
• The Arrow Flight server can handle multiple Parquet files concurrently, allowing for efficient loading of large datasets.
• The model_trainer.py script now uses the ArrowFlightDataset class, which is designed to connect to the arrow flight server, and load the data.
3. Training Setup:
• The training process is managed by the model_trainer.py script.
• The script uses PyTorch for model training and optimization.
• It supports checkpointing to save and resume training progress.
• It includes a WebSocket server to provide real-time training updates (loss, accuracy).
4. Hardware Setup:
• Training Server (Threadripper):
• 12-core Threadripper CPU.
• 128GB RAM.
• Two 24GB P40 GPUs.
• Six 8GB CMP 104-100 GPUs.
• Data Serving Server (Ryzen):
• 6-core Ryzen CPU.
• 100GB RAM.
• 8GB 4060 GPU.
• NVMe drive for data storage.
• Arrow Flight server to stream data.
5. Data Flow:
• The Ryzen machine serves pre-tokenized Parquet data via Arrow Flight.
• The Threadripper machine receives the data stream, loads it into PyTorch DataLoaders, and trains the model.
• The model_trainer.py script is designed to take the ip address, and port of the arrow flight server, as well as the list of parquet files to be used.
Current State:
• We have a robust data serving pipeline using Arrow Flight.
• The model architecture is defined and ready for training.
• The training script is set up to handle data loading, model training, and checkpointing.
• The project now utilizes the arrow flight server to load the data.
Next Steps:
• Begin the training process, monitoring performance and adjusting hyperparameters as needed.
• Implement evaluation metrics to assess the model's performance.
• Potentially investigate model parallelism to utilize all available GPUs.
