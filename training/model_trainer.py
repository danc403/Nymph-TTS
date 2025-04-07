#!/usr/bin/env python3
# model_trainer.py

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from tokenizers import Tokenizer
import argparse
import os
import asyncio
import websockets
import pyarrow.flight as fl
import pyarrow as pa
import io

# Load the tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

# Define a custom dataset using Arrow Flight
class ArrowFlightDataset(Dataset):
    def __init__(self, host, port, file_list):
        self.host = host
        self.port = port
        self.file_list = file_list
        self.table = self.fetch_data()
        self.data = self.table.to_pandas().to_dict('records') #convert to list of dicts.

    def fetch_data(self):
        location = fl.Location.for_grpc_tcp(self.host, self.port)
        client = fl.connect(location)
        ticket_str = ",".join(self.file_list)
        ticket = fl.Ticket(ticket_str.encode())
        reader = client.do_get(ticket)
        table = reader.read_all()
        return table

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_filepath = item["audio_filepath"]
        text = item["text"]
        speaker = item["speaker"]
        prosody = item["prosody"]
        sentiment = item["sentiment"]
        sample_rate = item["sample_rate"]
        mfccs = np.array(item["mfccs"])
        mel_spectrogram = np.array(item["mel_spectrogram"])

        # Load audio features
        audio, _ = librosa.load(audio_filepath, sr=sample_rate)
        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        # Tokenize text
        encoding = tokenizer.encode(text)
        input_ids = torch.tensor(encoding.ids, dtype=torch.long)
        attention_mask = torch.tensor(encoding.attention_mask, dtype=torch.long)

        # Convert prosody and sentiment to tensors
        prosody_tensor = torch.tensor([prosody["pitch"], prosody["energy"], prosody["rate"]], dtype=torch.float32)
        sentiment_tensor = torch.tensor([sentiment["score"]], dtype=torch.float32)

        # Convert MFCCs and Mel spectrograms to tensors
        mfccs_tensor = torch.tensor(mfccs, dtype=torch.float32)
        mel_spectrogram_tensor = torch.tensor(mel_spectrogram, dtype=torch.float32)

        # Convert speaker to integer (assuming speaker labels are in the format SPEAKER_01, SPEAKER_02, ...)
        speaker_id = int(speaker.split("_")[1])

        return {
            "audio": audio_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prosody": prosody_tensor,
            "sentiment": sentiment_tensor,
            "mfccs": mfccs_tensor,
            "mel_spectrogram": mel_spectrogram_tensor,
            "speaker": speaker_id
        }

# Custom Transformer
class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=4096, num_heads=32, num_layers=12, max_seq_length=512):
        super(CustomTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_length)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim * 4)
            for _ in range(num_layers)
        ])
        self.transformer = nn.TransformerEncoder(self.transformer_layers, num_layers)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        embedded = self.positional_encoding(embedded)
        embedded = embedded.transpose(0, 1)  # Transformer expects (seq_len, batch, features)
        output = self.transformer(embedded, src_key_padding_mask=~attention_mask.bool())
        output = output.transpose(0, 1)  # Back to (batch, seq_len, features)
        return output[:, 0, :]  # Return the CLS token equivalent.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Multimodal Model with Custom Transformer
class MultimodalModel(nn.Module):
    def __init__(self, num_speakers, vocab_size, latent_dim=1024, window_size=512, hidden_dim=4096, num_heads=32, num_layers=4, transformer_layers = 12, max_seq_length = 512):
        super(MultimodalModel, self).__init__()
        self.transformer = CustomTransformer(vocab_size, hidden_dim, num_heads, transformer_layers, max_seq_length) #custom transformer
        self.audio_linear = nn.Linear(161, hidden_dim)
        self.prosody_linear = nn.Linear(3, hidden_dim)
        self.sentiment_linear = nn.Linear(1, hidden_dim)
        self.speaker_embedding = nn.Embedding(num_speakers, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 4, hidden_dim)

        # MLA Layers
        self.key_compression = nn.Linear(hidden_dim, latent_dim)
        self.value_compression = nn.Linear(hidden_dim, latent_dim)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(latent_dim, num_heads) for _ in range(num_layers)
        ])
        self.latent_expansion = nn.Linear(latent_dim, hidden_dim)

        self.window_size = window_size

    def forward(self, audio, input_ids, attention_mask, prosody, sentiment, speaker):
        bert_output = self.transformer(input_ids, attention_mask) #use custom transformer
        audio_features = self.audio_linear(audio)
        audio_features = torch.mean(audio_features, dim=0)
        prosody_features = self.prosody_linear(prosody)
        sentiment_features = self.sentiment_linear(sentiment)
        speaker_features = self.speaker_embedding(speaker)

        keys = self.key_compression(bert_output)
        values = self.value_compression(bert_output)
        query = keys

        for attention_layer in self.attention_layers:
            attn_output, _ = attention_layer(query.transpose(0, 1), keys.transpose(0, 1), values.transpose(0, 1))
            query = attn_output.transpose(0, 1)

        bert_output = self.latent_expansion(query)

        if self.window_size > 0 and input_ids.size(1) > self.window_size:
            start = max(0, input_ids.size(1) - self.window_size)
            bert_output = bert_output[start:]

        combined_features = torch.cat([bert_output, audio_features, prosody_features, sentiment_features, speaker_features], dim=-1)
        output = self.fc(combined_features)
        return output

# Function to save the model and optimizer state
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

# Function to load the model and optimizer state
def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {checkpoint_path} at epoch {epoch} with loss {loss}")
    return model, optimizer, epoch, loss

async def send_data(websocket, epoch, loss, accuracy):
    data = {
        'epoch': epoch,
        'loss': loss,
        'accuracy': accuracy
    }
    await websocket.send(json.dumps(data))

async def main():
    parser = argparse.ArgumentParser(description="Train a multimodal LLM/TTS model.")
    parser.add_argument("--host", type=str, default="your_ryzen_ip", help="Arrow Flight server host.")
    parser.add_argument("--port", type=int, default=9090, help="Arrow Flight server port.")
    parser.add_argument("--file_list", nargs='+', default=["dataset1.parquet"], help="List of Parquet files to load.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Path to the checkpoint directory.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to the checkpoint file to resume training from.")
    parser.add_argument("--num_speakers", type=int, default=10, help="Number of speakers in the dataset.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--latent_dim", type=int, default=1024, help="Latent dimension for MLA.")
    parser.add_argument("--window_size", type=int, default=512, help="Window size for sliding window attention.")
    parser.add_argument("--hidden_dim", type=int, default=4096, help="Hidden dimension for linear layers.")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of MLA layers.")
    parser.add_argument("--transformer_layers", type=int, default=12, help="Number of transformer layers.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length for transformer.")

    args = parser.parse_args()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load dataset using Arrow Flight
    dataset = ArrowFlightDataset(args.host, args.port, args.file_list)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model, optimizer, and loss function
    vocab_size = tokenizer.get_vocab_size() #get vocab size from tokenizer
    model = MultimodalModel(args.num_speakers, vocab_size, args.latent_dim, args.window_size, args.hidden_dim, args.num_heads, args.num_layers, args.transformer_layers, args.max_seq_length)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()  # Using MSELoss for autoencoder approach

    # Load checkpoint if resuming training
    start_epoch = 0
    if args.resume_from:
        model, optimizer, start_epoch, _ = load_checkpoint(args.resume_from, model, optimizer)

    # Training loop
    async with websockets.serve(lambda websocket, path: training_loop(websocket, path, model, optimizer, criterion, dataloader, args.num_epochs, start_epoch, args.checkpoint_dir), "localhost", 8765):
        await asyncio.Future()  # run forever

async def training_loop(websocket, path, model, optimizer, criterion, dataloader, num_epochs, start_epoch, checkpoint_dir):
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in dataloader:
            optimizer.zero_grad()
            audio = batch["audio"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            prosody = batch["prosody"]
            sentiment = batch["sentiment"]
            speaker = batch["speaker"]

            # Autoencoder target: combined input features
            bert_output = model.transformer(input_ids, attention_mask)
            audio_features = model.audio_linear(audio)
            audio_features = torch.mean(audio_features, dim=0)
            prosody_features = model.prosody_linear(prosody)
            sentiment_features = model.sentiment_linear(sentiment)
            speaker_features = model.speaker_embedding(speaker)

            targets = torch.cat([bert_output, audio_features, prosody_features, sentiment_features, speaker_features], dim=-1)

            outputs = model(audio, input_ids, attention_mask, prosody, sentiment, speaker)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # Calculate accuracy (replace with your actual accuracy calculation)
            correct += 0 #Example accuracy calculation
            total += batch["audio"].size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0

        await send_data(websocket, epoch + 1, avg_loss, accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}, Accuracy: {accuracy}")
        save_checkpoint(model, optimizer, epoch + 1, avg_loss, checkpoint_dir)

if __name__ == "__main__":
    asyncio.run(main())
