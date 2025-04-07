#!/usr/bin/env python3
# main.py
import argparse
import os
from training.tokenizer_trainer import train_tokenizer
from training.transcriber import transcribe_audio

def main():
    parser = argparse.ArgumentParser(description="Train and use a tokenizer with JSONL input.")
    parser.add_argument("input_path", help="Path to the audio/text file or directory.")
    parser.add_argument("output_dir", help="Path to the output directory for JSONL files.")
    parser.add_argument("--tmp_dir", type=str, default="tmp", help="Path to the temporary directory.")
    parser.add_argument("--whisper_model", type=str, default="medium", help="Whisper model size (e.g., tiny, base, small, medium, large).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., cuda, cpu).")
    parser.add_argument("--no_diarize", action="store_false", dest="diarize", help="Disable speaker diarization.")
    parser.add_argument("--sentiment_model", type=str, default="roberta-base-go_emotions", help="Sentiment analysis model to use (e.g., roberta-base-go_emotions).")
    parser.add_argument("--sample_rate", type=int, help="Override the audio sample rate.")

    args = parser.parse_args()

    # Transcribe and create JSONL
    transcribe_audio(
        args.input_path,
        args.output_dir,
        args.tmp_dir,
        args.whisper_model,
        args.device,
        args.diarize,
        args.sentiment_model,
        args.sample_rate,
    )

    # Generate JSONL filename
    jsonl_filename = os.path.splitext(os.path.basename(args.input_path))[0] + ".jsonl"
    jsonl_filepath = os.path.join(args.output_dir, jsonl_filename)

    # Train tokenizer
    train_tokenizer(jsonl_filepath, "tokenizer_config.json")

if __name__ == "__main__":
    main()
