#!/usr/bin/env python3

import subprocess
import sys
import os
import importlib.util
import json
import time # Used for simulation; remove in actual model integration

# --- User Configuration ---
# IMPORTANT: Replace with your actual GitHub repository URL where voices are stored.
# This repository is expected to have a 'voices/' directory at its root,
# containing subdirectories for each voice (e.g., voices/my_voice_id/model.safetensors).
MY_GITHUB_REPO_URL = "https://github.com/your_github_user/your_tts_voices.git"

# Directory to store downloaded voice models locally
LOCAL_VOICES_DIR = "voices"

# List of essential Python packages this script requires.
# sounddevice is handled separately due to its system-level dependencies.
REQUIRED_PYTHON_PACKAGES = ["torch", "transformers", "huggingface_hub", "numpy"]
# --- End Configuration ---

# --- Dependency Check and Installation ---
def _check_and_install_package(package_name):
    """Checks if a Python package is installed and installs it if missing."""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"'{package_name}' not found. Attempting to install...")
        try:
            # Use sys.executable to ensure pip is called for the current Python interpreter
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"'{package_name}' installed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install '{package_name}'. Please try running 'pip install {package_name}' manually.")
            print(f"Details: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: An unexpected error occurred during installation of '{package_name}': {e}")
            sys.exit(1)
    else:
        print(f"'{package_name}' is already installed.")
        return True

def _check_and_install_all_deps():
    """Checks and installs all Python dependencies, including special handling for sounddevice."""
    print("\n--- Checking Python Dependencies ---")
    for pkg in REQUIRED_PYTHON_PACKAGES:
        _check_and_install_package(pkg)

    # Special check for sounddevice due to its reliance on system audio libraries
    try:
        import sounddevice as sd
        print("'sounddevice' is installed.")
        try:
            # Attempt to query devices to see if PortAudio backend is working
            sd.query_devices()
        except sd.PortAudioError:
            print("\nWARNING: 'sounddevice' installed, but PortAudio backend failed to initialize.")
            print("         This usually means you need to install system audio libraries.")
            print("         On Debian/Ubuntu: sudo apt-get install libportaudio2 python3-pyaudio")
            print("         On Fedora/RHEL: sudo dnf install portaudio-devel")
            print("         Audio output might not work until these are installed.")
    except ImportError:
        print("'sounddevice' not found. Attempting to install...")
        if _check_and_install_package("sounddevice"):
            try:
                import sounddevice as sd
                sd.query_devices() # Try initializing again
            except sd.PortAudioError:
                print("\nWARNING: 'sounddevice' installed, but PortAudio backend failed to initialize.")
                print("         You still need to install system audio libraries like PortAudio.")
                print("         See instructions above for your distribution.")
            except ImportError:
                print("ERROR: Failed to import 'sounddevice' even after attempting installation. Audio will not work.")
                sys.exit(1) # Critical failure if sounddevice can't be imported at all

# --- Voice Management ---
def _get_local_voices():
    """Returns a list of locally available voice IDs."""
    if not os.path.isdir(LOCAL_VOICES_DIR):
        return []
    available_voices = []
    # A voice is identified by a subdirectory containing a model file and config.json
    for voice_id in os.listdir(LOCAL_VOICES_DIR):
        voice_path = os.path.join(LOCAL_VOICES_DIR, voice_id)
        if os.path.isdir(voice_path):
            # Check for model file (pytorch_model.bin or model.safetensors) and config.json
            has_model = os.path.exists(os.path.join(voice_path, "pytorch_model.bin")) or \
                        os.path.exists(os.path.join(voice_path, "model.safetensors"))
            has_config = os.path.exists(os.path.join(voice_path, "config.json"))
            if has_model and has_config:
                available_voices.append(voice_id)
    return sorted(available_voices)

def _download_voice(voice_id):
    """
    Downloads a specific voice from the GitHub repository to the local voices directory.
    This function expects the GitHub repository to contain a 'voices/<voice_id>/'
    path that holds the model files.
    """
    print(f"Attempting to download voice '{voice_id}' from '{MY_GITHUB_REPO_URL}'...")
    local_voice_path = os.path.join(LOCAL_VOICES_DIR, voice_id)
    os.makedirs(local_voice_path, exist_ok=True)

    try:
        # Check if 'git' command is available
        print("Checking for 'git' command...")
        subprocess.check_call(["git", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("'git' is installed.")

        temp_clone_dir = f"temp_clone_for_voice_{voice_id}"
        if os.path.exists(temp_clone_dir):
            subprocess.run(["rm", "-rf", temp_clone_dir])

        print(f"Cloning temporary copy of '{MY_GITHUB_REPO_URL}' (shallow clone)...")
        # Use --depth 1 for shallow clone to save bandwidth and time
        # Suppress output to keep console clean
        subprocess.check_call(["git", "clone", "--depth", "1", MY_GITHUB_REPO_URL, temp_clone_dir],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        source_voice_path = os.path.join(temp_clone_dir, LOCAL_VOICES_DIR, voice_id)
        if not os.path.exists(source_voice_path):
            print(f"ERROR: Voice '{voice_id}' not found at expected path '{LOCAL_VOICES_DIR}/{voice_id}/' in the GitHub repository.")
            print("Please ensure the voice ID is correct and the repository structure is 'voices/<voice_id>/'.")
            subprocess.run(["rm", "-rf", temp_clone_dir])
            return False

        print(f"Moving voice files to '{local_voice_path}'...")
        # Copy contents to avoid issues with `mv` across filesystems or existing directories
        subprocess.check_call(["cp", "-r", f"{source_voice_path}/.", local_voice_path])

        subprocess.run(["rm", "-rf", temp_clone_dir]) # Clean up temporary clone
        print(f"Voice '{voice_id}' downloaded successfully to '{local_voice_path}'.")
        return True

    except FileNotFoundError:
        print("ERROR: 'git' command not found.")
        print("Please install Git on your system (e.g., 'sudo apt-get install git' on Debian/Ubuntu, or download from git-scm.com).")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Git command failed during voice download.")
        print(f"Details: {e}")
        print("Ensure the GitHub repository URL is correct and accessible.")
        if os.path.exists(temp_clone_dir):
            subprocess.run(["rm", "-rf", temp_clone_dir])
        return False
    except Exception as e:
        print(f"An unexpected error occurred during voice download: {e}")
        if os.path.exists(temp_clone_dir):
            subprocess.run(["rm", "-rf", temp_clone_dir])
        return False

# --- TTS Model Wrapper ---
# This class acts as an abstraction for your specific Qwen3-based TTS model.
# It handles loading and inference.
class MyTTSModel:
    def __init__(self, voice_path, device="cpu"):
        self.device = device
        print(f"Loading voice model from: {voice_path} to device: {self.device}...")
        try:
            from transformers import AutoTokenizer, AutoModelForSpeechSynthesis
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(voice_path)
            # Load your Qwen3-based TTS model. Replace AutoModelForSpeechSynthesis if you have a custom class name.
            # trust_remote_code=True might be necessary if your model's architecture requires it.
            # torch_dtype=torch.float16 is good for GPU if your model supports it for performance.
            self.model = AutoModelForSpeechSynthesis.from_pretrained(
                voice_path,
                torch_dtype=torch.float16 if "cuda" in device else torch.float32,
                device_map="auto" if "cuda" in device else None, # Leverages accelerate for multi-GPU or efficient loading
                local_files_only=True # Assumes voice_path contains all necessary files
            ).to(device)
            # Set model to evaluation mode for inference
            self.model.eval()

            # Attempt to get sample rate from model config, default if not found
            self.sample_rate = getattr(self.model.config, "sampling_rate", 22050)
            print(f"Model loaded. Detected sample rate: {self.sample_rate} Hz.")
        except ImportError:
            print("ERROR: 'torch' or 'transformers' not found. Please ensure all dependencies are installed.")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Could not load voice model from '{voice_path}'.")
            print(f"Please ensure the directory contains 'config.json' and model weights (pytorch_model.bin or model.safetensors).")
            print(f"Details: {e}")
            sys.exit(1)

    def synthesize_stream(self, text, speed_rate=1.0):
        """
        Synthesizes text to speech and yields audio chunks.
        The speed_rate parameter is passed to the model's generate method.
        """
        import torch
        import numpy as np

        print(f"Synthesizing text with speed rate: {speed_rate}...")

        # Basic text chunking for streaming. For production, consider proper sentence tokenization.
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            print("No valid text to synthesize.")
            return

        with torch.no_grad(): # Disable gradient calculations for inference
            for i, sentence in enumerate(sentences):
                # Tokenize the input sentence
                inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)

                # --- Replace this with your actual Qwen3-based model's generation logic ---
                # Your custom model's 'generate' method should accept 'speed' and ideally 'stream=True'
                # and yield audio chunks.
                # Example for a hypothetical model that yields chunks:
                # audio_chunks_generator = self.model.generate(
                #     **inputs,
                #     speed=speed_rate, # Pass speed rate to your model's generation
                #     stream=True,     # Indicate streaming mode if your model supports it
                #     return_dict_in_generate=True,
                #     output_attentions=False,
                #     output_hidden_states=False,
                #     return_tensors="pt"
                # )
                # for output_chunk in audio_chunks_generator:
                #     yield output_chunk['audio_chunk'].cpu().numpy().flatten()
                # --- End Model-Specific Logic ---

                # --- SIMULATION ONLY (REMOVE IN REAL IMPLEMENTATION) ---
                print(f"  [Simulating] Processing sentence {i+1}/{len(sentences)}: '{sentence}'...")
                # Simulate a chunk of audio based on sentence length and speed
                duration_per_char = 0.08 / speed_rate # Shorter duration for faster speech
                num_samples = int(len(sentence) * duration_per_char * self.sample_rate)
                t = np.linspace(0, len(sentence) * duration_per_char, num_samples, endpoint=False)
                # Simple sine wave for simulation; your model will produce actual speech
                simulated_audio_chunk = np.sin(440 * 2 * np.pi * t) * 0.5 + np.sin(880 * 2 * np.pi * t) * 0.25
                simulated_audio_chunk = simulated_audio_chunk.astype(np.float32) # Ensure float32

                yield simulated_audio_chunk
                time.sleep(len(sentence) * duration_per_char / 5) # Simulate processing time
                # --- END SIMULATION ---

# --- Main Application Entry Point ---
def run_tts_engine():
    # 1. Check and install Python dependencies (this will exit if critical deps fail)
    _check_and_install_all_deps()

    # Import modules after successful dependency check
    import argparse
    import torch
    import sounddevice as sd
    import numpy as np # Used in simulation and for audio array manipulation

    print("\n--- Initializing TTS Engine ---")
    parser = argparse.ArgumentParser(
        description="Simple Streaming TTS Inference Engine.",
        formatter_class=argparse.RawTextHelpFormatter # Preserve newlines in help
    )
    parser.add_argument("--text", type=str, help="Text to convert to speech. If omitted, prompts user interactively.")
    parser.add_argument(
        "--voice",
        type=str,
        default="default", # A default voice ID
        help="Voice ID to use. E.g., 'male1', 'female2'.\n"
             "The script will check local voices first. If not found,\n"
             "it will offer to download from your GitHub repository."
    )
    parser.add_argument("--rate", type=float, default=1.0, help="Speaking rate (e.g., 0.8 for slower, 1.2 for faster). Default is 1.0.")
    args = parser.parse_args()

    # Determine execution device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU.")

    # 2. Voice Selection and Loading
    available_voices = _get_local_voices()
    print(f"\nCurrently available local voices: {available_voices if available_voices else 'None'}")

    chosen_voice_id = args.voice
    voice_path = os.path.join(LOCAL_VOICES_DIR, chosen_voice_id)

    if chosen_voice_id not in available_voices:
        print(f"Voice '{chosen_voice_id}' not found locally.")
        user_response = input(f"Do you want to download voice '{chosen_voice_id}' from GitHub ('{MY_GITHUB_REPO_URL}')? (y/N): ").lower()
        if user_response == 'y':
            if not _download_voice(chosen_voice_id):
                print(f"Failed to download voice '{chosen_voice_id}'. Exiting.")
                sys.exit(1)
            # Re-check local voices after download attempt
            available_voices = _get_local_voices()
            if chosen_voice_id not in available_voices:
                print(f"Voice '{chosen_voice_id}' is still not available after download attempt. Please check logs. Exiting.")
                sys.exit(1)
        else:
            print("Voice download cancelled. Exiting.")
            sys.exit(0)
    else:
        print(f"Using locally available voice: '{chosen_voice_id}'.")


    # 3. Load the chosen voice model
    tts_model = MyTTSModel(voice_path, device=device) # Instantiate your custom model wrapper

    # 4. Get text input
    text_to_synthesize = args.text
    if not text_to_synthesize:
        print("\nEnter text to synthesize. Type 'quit' or 'exit' to stop.")
        while True:
            text_input = input("Text: ").strip()
            if text_input.lower() in ['quit', 'exit']:
                break
            if not text_input:
                continue

            _stream_and_play_audio(tts_model, text_input, args.rate)
    else:
        _stream_and_play_audio(tts_model, text_to_synthesize, args.rate)

    print("\nScript finished.")

def _stream_and_play_audio(tts_model, text, rate):
    """Handles the audio streaming and playback."""
    import sounddevice as sd
    import numpy as np

    print("\n--- Playing audio ---")
    try:
        # Get sample rate from the loaded model
        sample_rate = tts_model.sample_rate

        # Open an audio stream for playback
        audio_stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32')
        audio_stream.start()

        for audio_chunk in tts_model.synthesize_stream(text, rate):
            # Ensure chunk is a 1D numpy array of float32
            if not isinstance(audio_chunk, np.ndarray):
                print("Warning: Received non-numpy array audio chunk. Skipping.")
                continue
            if audio_chunk.ndim > 1:
                audio_chunk = audio_chunk.flatten() # Flatten to mono if stereo/multi-channel
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)

            audio_stream.write(audio_chunk)

        audio_stream.stop()
        audio_stream.close()
        print("\nFinished playing audio.")

    except sd.PortAudioError as e:
        print(f"AUDIO ERROR: Sound output failed. This often means PortAudio (a system library) is not installed or configured correctly.")
        print(f"Details: {e}")
        print("Please check previous warnings about system dependencies.")
    except Exception as e:
        print(f"An unexpected error occurred during audio playback: {e}")
        # Ensure stream is closed in case of error
        if 'audio_stream' in locals() and audio_stream.active:
            audio_stream.stop()
            audio_stream.close()


if __name__ == "__main__":
    run_tts_engine()
