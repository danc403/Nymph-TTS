#!/usr/bin/env python3
#File: `tokenizer_trainer.py`

import argparse
import os
import json
import nltk
from typing import List, Dict, Optional, Iterator
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from lxml import etree

from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure nltk is installed and punkt is downloaded
try:
    from nltk.tokenize import sent_tokenize
except ImportError:
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError:
    import nltk
    nltk.download('vader_lexicon')
    from nltk.sentiment import SentimentIntensityAnalyzer

import librosa
import librosa.display
import numpy as np

def train_tokenizer(
    jsonl_file: str,  # Changed to accept the jsonl file
    config_file: str,
) -> callable:
    """
    Trains a tokenizer based on the provided JSONL file and configuration.
    Uses Hugging Face `tokenizers` library.
    If audio_dir is provided, it processes audio and adds SSML markup to the text.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)

    vocab_size = config.get("vocab_size", 1000)
    special_tokens = config.get("special_tokens", ["<unk>", "<pad>", "<bos>", "<eos>"])
    context_tokens = config.get("context_tokens", ["[CONTEXT_PREV_SENT]"])
    oov_token = config.get("oov_token", "<unk>")
    context_window_size = config.get("context_window_size", 3)  # Default context window size
    emotion_labels = config.get("emotion_labels", [])

    # Initialize the tokenizer
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()  # Simple whitespace pre-tokenizer

    # Initialize the trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens + context_tokens,
        show_progress = True
    )

    # Train the tokenizer
    def text_generator():
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data["text"]
                    clip_name = data["clip_name"]
                    speaker = data["speaker"]
                    prosody = data["prosody"]
                    sentiment = data["sentiment"]
                    audio_filepath = data["audio_filepath"]
                    start_time = data["start_time"]
                    end_time = data["end_time"]
                    sample_rate = data["sample_rate"]
                    mfccs = data.get("mfccs") # Get MFCC data

                    # Create emotion/sentiment tokens based on sentiment scores
                    sentiment_token = ""
                    if sentiment:
                        sentiment_label = sentiment.get("label")  # Get sentiment label

                        # Map sentiment labels to tokens, ensuring consistency
                        if sentiment_label in emotion_labels:
                            sentiment_token = f"[EMO_{sentiment_label.upper()}]"
                        else:
                            sentiment_token = "[EMO_UNKNOWN]"  # Handle unknown labels
                        # Add speaker token
                    speaker_token = f"[SPEAKER_{speaker}]"

                    # Create prosody tokens
                    pitch_token = f"[PITCH_{prosody['pitch']:.2f}]"
                    energy_token = f"[ENERGY_{prosody['energy']:.2f}]"
                    rate_token = f"[RATE_{prosody['rate']:.2f}]"

                    # Add Start and End time tokens for audio sync
                    start_token = f"[START_TIME_{start_time:.2f}]"
                    end_token = f"[END_TIME_{end_time:.2f}]"

                    #Create tokens for mfccs
                    mfcc_tokens = []
                    if mfccs:
                        # Discretize MFCC values (example with 5 bins)
                        num_bins = 5
                        min_mfcc = min(min(mfcc) for mfcc in mfccs)
                        max_mfcc = max(max(mfcc) for mfcc in mfccs)
                        bins = np.linspace(min_mfcc, max_mfcc, num_bins + 1)

                        for i, mfcc_values in enumerate(mfccs):
                            for j, value in enumerate(mfcc_values):
                                # Find the bin
                                bin_index = np.digitize(value, bins) - 1  # Bins are 1-indexed
                                mfcc_tokens.append(f"[MFCC_{i}_{j}_BIN{bin_index}]")  # example: [MFCC_3_4_BIN2]

                    #Combine all tokens with the text
                    combined_text = f"{sentiment_token} {speaker_token} {pitch_token} {energy_token} {rate_token} {start_token} {end_token} {' '.join(mfcc_tokens)} {text}"

                    yield combined_text
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                except KeyError as e:
                    print(f"Missing key in JSON: {e}")

    tokenizer.train_from_iterator(text_generator(), trainer=trainer)

    # Save the tokenizer
    tokenizer.save("tokenizer.json")
    print("Tokenizer saved to tokenizer.json")

    # Context Handling Function
    def create_context_window(sentences: List[str], index: int, window_size: int) -> str:
        """Creates a context window around a sentence."""
        start = max(0, index - window_size)
        end = min(len(sentences), index + window_size + 1)
        context = " ".join(sentences[start:index] + sentences[index+1:end]) # Exclude current sentence
        return context

    def insert_punctuation_tokens(text: str) -> str:
      """Inserts punctuation tokens based on sentence endings."""
      if text.endswith("?"):
          return f"[PUNCT_QUESTION] {text}"
      elif text.endswith("!"):
          return f"[PUNCT_EXCLAMATION] {text}"
      else:
          return text  # Return original text if no specific punctuation detected

    def tokenize(text: str) -> List[str]:
        """Tokenizes the input text."""
        all_tokens = []

        # Split text into sentences
        sentences = sent_tokenize(text)

        for i, sentence in enumerate(sentences):

            # Insert Punctuation tokens
            sentence = insert_punctuation_tokens(sentence)

            # Add Context tokens
            context = create_context_window(sentences, i, context_window_size)
            sentence_with_context = "[CONTEXT_PREV_SENT] " + context + " " + sentence  # Combine context and sentence

            # Tokenize using the trained tokenizer
            encoding = tokenizer.encode(sentence_with_context)  # Encode entire string with the sentence and context.
            tokens = encoding.tokens  # get the tokens out.

            all_tokens.extend(tokens)  # Add to the final list of tokens

        return all_tokens

    return tokenize
