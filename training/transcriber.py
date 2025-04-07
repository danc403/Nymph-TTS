#!/usr/bin/env python3
#File: `transcriber.py`

import argparse
import os
import json
import librosa
import librosa.display
import numpy as np
import whisperx
import soundfile as sf
import torch
from typing import List, Dict, Tuple, Optional
from transformers import pipeline  # For transformer sentiment
import tempfile  # For temporary directories
import nltk

# MFCC dependencies
import numpy as np
import librosa

def extract_mfccs(audio_path: str, sr: int = None, n_mfcc: int = 40) -> np.ndarray:
    """Extracts MFCC features from the audio file."""
    y, sr = librosa.load(audio_path, sr=sr)  # Load using the given sample rate. If None librosa will use native SR
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs

def extract_mel_spectrogram(audio_path: str, sr: int = None) -> np.ndarray:
    """Extracts mel spectrogram features from the audio file."""
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    return mel_spectrogram

def extract_prosody_features(audio_path: str, sr: int = None) -> Dict[str, float]:  # Added SR
    """Extracts prosody features (pitch, rate, energy) from the audio file using librosa."""
    try:
        y, sr = librosa.load(audio_path, sr=sr)  # Load audio with the given or native sample rate

        # Pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = np.mean(pitches[magnitudes > 0])

        # Energy (RMS)
        rms = librosa.feature.rms(y=y)[0]
        energy = np.mean(rms)

        # Rate (Tempo)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        return {"pitch": pitch, "energy": energy, "rate": tempo}

    except Exception as e:
        print(f"Error extracting prosody features: {e}")
        return {"pitch": 0.0, "energy": 0.0, "rate": 0.0}

def analyze_sentiment(text: str, sentiment_pipeline) -> Dict[str, float]:
    """Analyzes sentiment using a transformer model, handling multiple labels."""
    try:
        result = sentiment_pipeline(text, truncation=True, max_length=512)  # Truncate if needed

        # Handle single-label results
        if isinstance(result, dict) and "label" in result and "score" in result:
            return {"label": result["label"], "score": result["score"]}

        # Handle multi-label results (if the model returns a list of dicts)
        elif isinstance(result, list) and all(isinstance(item, dict) and "label" in item and "score" in item for item in result):
            # Return the label with the highest score
            best_label = max(result, key=lambda x: x["score"])
            return {"label": best_label["label"], "score": best_label["score"]}

        else:
            print(f"Unexpected sentiment analysis result format: {result}")
            return {"label": "UNKNOWN", "score": 0.0}  # Return a default value in case of error

    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return {"label": "ERROR", "score": 0.0}  # Return a default value in case of error

def process_audio(audio_path: str, output_dir: str, tmp_dir: str, whisper_model: str, device: str, diarize: bool, sentiment_pipeline, sample_rate: Optional[int]) -> None:
    """Processes audio files."""

    os.makedirs(tmp_dir, exist_ok=True)

    model = whisperx.load_model(whisper_model, device)

    audio_info = sf.SoundFile(audio_path)
    native_sr = audio_info.samplerate

    sr = sample_rate if sample_rate is not None else native_sr

    audio = whisperx.load_audio(audio_path, sr=sr)

    result = model.transcribe(audio, batch_size=16)

    model_a, metadata = whisperx.load_align_model(language=result["language"], device=device)
    result = whisperx.align(result["segments"], audio, model_a, metadata, device, return_char_alignments=False)

    if diarize:
        diarize_model = whisperx.DiarizationModel(use_auth_token=False, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_speaker_labels(diarize_segments, result)

    jsonl_filename = os.path.splitext(os.path.basename(audio_path))[0] + ".jsonl"
    jsonl_filepath = os.path.join(output_dir, jsonl_filename)

    with open(jsonl_filepath, "a", encoding="utf-8") as f:
        clip_counter = 0
        for segment in result["segments"]:
            text = segment["text"].strip()
            start = segment["start"]
            end = segment["end"]
            speaker = segment.get("speaker", "SPEAKER_00")
            duration = end - start

            sentiment_scores = analyze_sentiment(text, sentiment_pipeline)

            clip_name = f"{os.path.splitext(os.path.basename(audio_path))[0]}-{clip_counter:04d}.wav"

            clip_start_sample = int(start * sr)
            clip_end_sample = int(end * sr)
            audio_clip = audio[clip_start_sample:clip_end_sample]

            clip_filepath = os.path.join(tmp_dir, clip_name)
            sf.write(clip_filepath, audio_clip, sr)

            clip_mfccs = extract_mfccs(clip_filepath, sr=sr)
            clip_mel_spectrogram = extract_mel_spectrogram(clip_filepath, sr=sr)

            prosody_features = extract_prosody_features(clip_filepath, sr=sr)

            jsonl_entry = {
                "audio_filepath": clip_filepath,
                "clip_name": clip_name,
                "start_time": start,
                "end_time": end,
                "text": text,
                "speaker": speaker,
                "prosody": prosody_features,
                "sentiment": sentiment_scores,
                "sample_rate": sr,
                "mfccs": clip_mfccs.tolist() if clip_mfccs is not None else None,
                "mel_spectrogram": clip_mel_spectrogram.tolist() if clip_mel_spectrogram is not None else None,
            }
            json.dump(jsonl_entry, f, ensure_ascii=False)
            f.write("\n")
            clip_counter += 1

    print(f"Transcription, feature extraction, and JSONL creation complete for: {audio_path}")

def process_text_file(input_path: str, output_dir: str, sentiment_pipeline) -> None:
    """Processes text files."""
    jsonl_filename = os.path.splitext(os.path.basename(input_path))[0] + ".jsonl"
    jsonl_filepath = os.path.join(output_dir, jsonl_filename)

    with open(input_path, "r", encoding="utf-8") as text_file, \
         open(jsonl_filepath, "a", encoding="utf-8") as jsonl_file:

        text = text_file.read()
        sentences = nltk.sent_tokenize(text)

        for sentence in sentences:
            sentiment = analyze_sentiment(sentence, sentiment_pipeline)
            jsonl_entry = {
                "text": sentence,
                "audio_filepath": None,
                "start_time": None,
                "end_time": None,
                "prosody": None,
                "mfccs": None,
                "mel_spectrogram": None,
                "sentiment": sentiment,
            }
            json.dump(jsonl_entry, jsonl_file, ensure_ascii=False)
            jsonl_file.write("\n")

def process_external_jsonl(input_path: str, output_dir: str, sentiment_pipeline) -> None:
    """Processes external JSONL files."""
    jsonl_filename = os.path.splitext(os.path.basename(input_path))[0] + "_processed.jsonl"
    jsonl_filepath = os.path.join(output_dir, jsonl_filename)

    with open(input_path, "r", encoding="utf-8") as in_file, \
         open(jsonl_filepath, "a", encoding="utf-8") as out_file:

        for line in in_file:
            try:
                entry = json.loads(line)
                if "audio_filepath" in entry or "clip_path" in entry or "audio_clip" in entry :
                    # process audio data
                    #TODO add code for processing external jsonl files with audio
                    print("external jsonl file with audio found. audio processing not implemented yet.")
                else:
                    sentiment = analyze_sentiment(entry["text"], sentiment_pipeline)
                    new_entry = {
                        "text": entry["text"],
                        "audio_filepath": None,
                        "start_time": None,
                        "end_time": None,
                        "prosody": None,
                        "mfccs": None,
                        "mel_spectrogram": None,
                        "sentiment": sentiment,
                    }
                json.dump(new_entry, out_file, ensure_ascii=False)
                out_file.write("\n")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

def transcribe_audio(input_path: str, output_dir: str, tmp_dir: str, whisper_model: str, device: str, diarize: bool, sentiment_model: str, sample_rate: Optional[int]) -> None:
    """Transcribes audio or text, extracts features, and creates a JSONL file."""

    sentiment_pipeline = pipeline("text-classification", model=sentiment_model, device=device)

    if os.path.isfile(input_path):
        if input_path.endswith((".wav", ".mp3", ".ogg", ".flac")):
            process_audio(input_path, output_dir, tmp_dir, whisper_model, device, diarize, sentiment_pipeline, sample_rate)
        elif input_path.endswith(".txt"):
            process_text_file(input_path, output_dir, sentiment_pipeline)
        elif input_path.endswith(".jsonl"):
            process_external_jsonl(input_path, output_dir, sentiment_pipeline)
        else:
            print(f"Unsupported file type: {input_path}")
    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            file_path = os.path.join(input_path, filename)
            if os.path.isfile(file_path):
                transcribe_audio(file_path, output_dir, tmp_dir, whisper_model, device, diarize, sentiment_model, sample_rate)
    else:
        print(f"Invalid input path: {input_path}")

def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Transcribe audio, extract features, and generate JSONL.")
    parser.add_argument("input_path", help="Path to the audio/text file or directory.")
    parser.add_argument("output_dir", help="Path to the output directory for JSONL files.")
    parser.add_argument("--tmp_dir", type=str, default="tmp", help="Path to the temporary directory.")
    parser.add_argument("--whisper_model", type=str, default="medium", help="Whisper model size (e.g., tiny, base, small, medium, large).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., cuda, cpu).")
    parser.add_argument("--no_diarize", action="store_false", dest="diarize", help="Disable speaker diarization.")
    parser.add_argument("--sentiment_model", type=str, default="roberta-base-go_emotions", help="Sentiment analysis model to use (e.g., roberta-base-go_emotions).")
    parser.add_argument("--sample_rate", type=int, help="Override the audio sample rate.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    transcribe_audio(args.input_path, args.output_dir, args.tmp_dir, args.whisper_model, args.device, args.diarize, args.sentiment_model, args.sample_rate)

if __name__ == "__main__":
    main()
