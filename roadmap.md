Project Roadmap: High-Performance, Customizable TTS for Linux Accessibility
1. Overall Project Goal:
• To develop a Text-To-Speech (TTS) solution for Linux that provides high-quality, extremely low-latency, and natural-sounding speech output, suitable for full-time use by a screen reader user (specifically for programming and other demanding tasks).
• The primary integration target is the Speech Dispatcher (SPD) framework, making it accessible to screen readers like Orca.
• The aim is to achieve responsiveness and clarity comparable to preferred Windows-based TTS voices (e.g., Windows OneCore "David"), addressing the current limitations of existing Linux TTS options for power users.
2. Core TTS Model Strategy:
• Model Architecture: Train a custom, end-to-end Transformer-based TTS model. 
• This model will directly output raw PCM audio data, eliminating the need for a separate vocoder step at inference time, which simplifies the runtime module and reduces a potential latency source.
• Model Size & Optimization:
• Initial target size: Approximately 0.5 billion parameters as a starting point for achieving high quality.
• Iterative Reduction: Systematically work to reduce model size (e.g., towards a range like 60M-150M parameters or as low as possible) while meticulously evaluating the trade-off to find an acceptable medium between voice quality, naturalness, and inference speed.
• Quantization: Heavily leverage model quantization techniques to significantly reduce model footprint and accelerate inference on CPU.
• Text Preprocessing:
• The end-to-end model itself will be trained to handle most text normalization and phonemization internally. This means it should learn to process raw or nearly raw text input.
• This simplifies the external C/C++ workflow in the runtime module but places a strong emphasis on the diversity and quality of training data to cover numbers, abbreviations, special characters, etc.
• Training Data:
• Primary Dataset: Several hundred hours of high-quality prose/literature audio (e.g., at 48kHz sampling rate) will form the base for achieving high acoustic quality. 
• Utilize tools like Whisper and custom scripts for accurate diarization and segmentation of this audio data.
• Supplemental Data: Gradually mix in conversational speech data in smaller amounts during training to improve naturalness and adaptability for a wider range of contexts.
3. Specialization and Voice Customization:
• LoRA (Low-Rank Adaptation):
• Employ LoRA to adapt the high-quality base prose model for specialized domains (e.g., "code reading voice," "financial news voice") without requiring full retraining of the large base model.
• This allows for parameter-efficient fine-tuning.
• Model Format for LoRA:
• Primarily target the .gguf model format due to its flexibility and strong LoRA support within the ggml ecosystem (e.g., as demonstrated by llama.cpp functionalities).
• Dynamic Adapter Loading: The Speech Dispatcher module should be capable of loading the base .gguf model and then applying different .gguf-formatted LoRA adapters based on the "voice" selected by the user through SPD.
4. Inference Engine Strategy (within the C/C++ SPD Module):
• The core inference will happen in C/C++ for maximum performance.
• Primary Option: .gguf with ggml-based C/C++ Code:
• Leverage the ggml C library to write custom C/C++ code that loads the .gguf base model and selected LoRA adapters.
• This C/C++ code will implement the Transformer model's forward pass (inference logic) using ggml's tensor operations.
• Alternative Option: .onnx with ONNX Runtime:
• Convert the trained model to .onnx format.
• Use the ONNX Runtime C/C++ API within the SPD module to load and execute the model. This is a mature, highly optimized runtime.
• Both options will work with the quantized version of your model.
5. Speech Dispatcher (SPD) Integration (.so Module):
• Native C/C++ Module: The TTS engine will be packaged as a native Linux shared library (.so file) that Speech Dispatcher can load directly. This is critical for achieving the lowest possible communication latency.
• SPD C API Implementation: The .so module must correctly implement all necessary C functions defined by the Speech Dispatcher module interface. Key functions include: 
• spd_open(): Initialize engine, load models/adapters.
• spd_close(): Release resources.
• spd_list_voices(): Present the base voice and different LoRA-adapted versions (e.g., "MyTTS-Prose", "MyTTS-Code") as distinct selectable voices.
• spd_set_voice(): Load/apply the appropriate base model and LoRA adapter.
• spd_synth_text(): Receive text from SPD, perform inference using the C/C++ engine, and provide raw PCM audio data back to SPD.
• spd_cancel(): Stop ongoing speech.
• Functions for setting rate, pitch, volume (if the model/engine supports runtime adjustment).
• Direct Audio Output: The module will pass the raw PCM audio data (generated directly by the end-to-end model) back to SPD, which then handles playback.
6. Development Workflow & Prototyping:
• Python's Role:
• Core ML Work: All model training, experimentation, data preparation, and initial phonemizer development (even if the final model internalizes it) will be done in Python.
• Prototyping Inference Logic: Use Python (with Python bindings for ggml or ONNX Runtime if available) to prototype the inference pipeline (text input -> tensor creation -> model execution -> audio output). This helps validate the model and the data flow before writing C++.
• C/C++ Implementation:
• The actual SPD .so module must be written in C or C++.
• The Python prototype will serve as a detailed blueprint for this C++ implementation.
• The process involves manually translating the logic from the Python prototype into C++, using C++ libraries and the C APIs of SPD and the chosen C/C++ inference engine.
• Initial C++ Goal:
• Focus on creating a functional, working C++ .so module, even if it's not perfectly optimized initially.
• This demonstrates viability and provides a concrete C++ codebase that can attract collaboration from more experienced C++ developers for further optimization and refinement.
• AI Assistance (Your collaboration with me):
• Use AI assistance to clarify C++ concepts, get suggestions for mapping Python patterns to C++, identify equivalent C++ libraries, review C++ snippets for logic, and understand the SPD C API. The AI will not write the full C++ module or debug complex runtime issues but can act as a knowledgeable consultant for specific C++ questions during the reimplementation.
7. Key Technical Decisions & Rejected Alternatives (for context):
• End-to-End Model (Raw Audio Out): Chosen to simplify the runtime module.
• Internalized Preprocessing: Model aims to handle normalization/phonemization to reduce external dependencies in C++.
• .so Module for SPD: Non-negotiable for low latency.
• C/C++ for .so Module: Mandatory for SPD integration and performance.
• Avoided:
• Direct porting of Windows TTS components (licensing/technical infeasibility).
• HTTP-based APIs between SPD and TTS engine (latency).
• stdin/stdout wrappers for SPD (found to still have too much latency by user).
• Writing SPD modules directly in Python (not supported by SPD for native modules).
