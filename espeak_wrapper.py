import subprocess
import numpy as np
from scipy.signal import butter, lfilter
import soundfile as sf # Used for potential future file operations, kept for consistency
import io
import sys
import os

# --- Dependency Management ---

def check_command(command):
    """Checks if a command exists in the system's PATH."""
    try:
        subprocess.run([command, "--version"], capture_output=True, check=True,
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_python_module(module_name):
    """Checks if a Python module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def get_package_manager():
    """Detects the system's package manager."""
    if check_command("apt"):
        return "apt"
    elif check_command("dnf"):
        return "dnf"
    elif check_command("yum"): # For older RHEL/CentOS
        return "yum"
    else:
        return None

def install_system_package(package_manager, package_name_apt, package_name_dnf):
    """Installs a system package using sudo."""
    print(f"\n--- Installing system package: {package_name_apt if package_manager == 'apt' else package_name_dnf} ---")
    if package_manager == "apt":
        install_command = ["sudo", "apt", "update", "-y"] # Update first
        try:
            print("Running: sudo apt update -y")
            subprocess.run(install_command, check=True)
        except subprocess.CalledProcessError:
            print("Failed to update apt package lists. Proceeding without update, but installation might fail.")

        install_command = ["sudo", "apt", "install", "-y", package_name_apt]
    elif package_manager in ["dnf", "yum"]:
        install_command = ["sudo", package_manager, "install", "-y", package_name_dnf]
    else:
        print("Could not determine package manager (apt/dnf/yum). Please install manually.")
        return False

    print(f"Running: {' '.join(install_command)}")
    try:
        subprocess.run(install_command, check=True)
        print(f"Successfully installed {package_name_apt if package_manager == 'apt' else package_name_dnf}.")
        return True
    except subprocess.CalledProcessError:
        print(f"Error installing {package_name_apt if package_manager == 'apt' else package_name_dnf}. Please check your sudo password and try again, or install manually.")
        return False

def install_python_package(package_name):
    """Installs a Python package using pip, trying sudo then --user if needed."""
    print(f"\n--- Installing Python package: {package_name} ---")
    # Try system-wide install first (may require sudo or be done from virtualenv)
    try:
        print(f"Attempting system-wide pip install for {package_name}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package_name], check=True)
        print(f"Successfully installed {package_name}.")
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package_name} system-wide. Trying with '--user' option...")
        # Fallback to user-specific install
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--user", package_name], check=True)
            print(f"Successfully installed {package_name} (user install).")
            return True
        except subprocess.CalledProcessError:
            print(f"Error installing {package_name} even with --user. Please install manually:")
            print(f"  {sys.executable} -m pip install {package_name}")
            print(f"  or {sys.executable} -m pip install --user {package_name}")
            return False

def ensure_dependencies():
    """
    Ensures all necessary system and Python dependencies are installed.
    Asks for user confirmation before installing.
    """
    deps_missing = []

    # Check for espeak-ng (critical system command)
    if not check_command("espeak-ng"):
        deps_missing.append(("system", "espeak-ng", "espeak-ng"))

    # Check for portaudio development headers (critical for pyaudio build)
    # We can't directly check for dev headers, so we'll just queue it if PyAudio is missing.
    # We need to detect PyAudio first, then suggest portaudio-dev if PyAudio is missing.
    if not check_python_module("pyaudio"):
        pkg_manager = get_package_manager()
        if pkg_manager == "apt":
            deps_missing.append(("system", "portaudio19-dev", "portaudio19-dev"))
        elif pkg_manager in ["dnf", "yum"]:
            deps_missing.append(("system", "portaudio-devel", "portaudio-devel"))
        else:
            print("Warning: Cannot determine package manager to suggest PortAudio development headers. PyAudio might fail to install.")


    # Check for Python modules
    python_modules_to_check = ["pyaudio", "numpy", "scipy", "soundfile"]
    for module in python_modules_to_check:
        if not check_python_module(module):
            deps_missing.append(("python", module))

    if not deps_missing:
        print("All required dependencies are already installed.")
        return True

    print("\n--- Detected Missing Dependencies ---")
    for dep_type, *dep_info in deps_missing:
        if dep_type == "system":
            print(f"- System package: {dep_info[0]} (for Debian/Ubuntu) / {dep_info[1]} (for Rocky/RHEL)")
        else: # python
            print(f"- Python module: {dep_info[0]}")

    response = input("\nSome dependencies are missing. Would you like to attempt to install them now? (y/N): ").strip().lower()
    if response != 'y':
        print("Installation cancelled. Cannot proceed without dependencies.")
        return False

    pkg_manager = get_package_manager()
    if not pkg_manager and any(d[0] == "system" for d in deps_missing):
        print("\nError: Cannot detect a supported system package manager (apt/dnf/yum).")
        print("Please manually install the listed system packages.")
        return False

    success = True
    for dep_type, *dep_info in deps_missing:
        if dep_type == "system":
            if not install_system_package(pkg_manager, dep_info[0], dep_info[1]):
                success = False
        else: # python
            if not install_python_package(dep_info[0]):
                success = False

    if not success:
        print("\nFailed to install all dependencies. Please resolve any errors above and try again.")
        sys.exit(1) # Exit if critical dependencies couldn't be installed
    
    # Re-import PyAudio after potential installation
    try:
        import pyaudio
        global pyaudio # Make it available globally if needed by other functions
        print("PyAudio module successfully imported after installation attempt.")
    except ImportError:
        print("Error: PyAudio still not found after installation attempt. Cannot proceed.")
        sys.exit(1)


    return True

# --- Audio Processing and Playback (from previous script) ---

# Global PyAudio object, will be initialized after dependencies are confirmed
p_audio_instance = None

def get_pyaudio_instance():
    """Returns a PyAudio instance, initializing it if necessary."""
    global p_audio_instance
    if p_audio_instance is None:
        try:
            import pyaudio
            p_audio_instance = pyaudio.PyAudio()
        except ImportError:
            print("PyAudio is not installed. Cannot play audio.", file=sys.stderr)
            sys.exit(1) # Exit if PyAudio somehow isn't available after checks
    return p_audio_instance


def apply_lowpass_filter(data, cutoff_freq, sample_rate, order=5):
    """
    Applies a Butterworth low-pass filter to the audio data.
    Helps reduce harsh high frequencies.
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def normalize_audio(audio_data, target_rms=0.1):
    """
    Normalizes audio data to a target RMS level.
    Prevents clipping and ensures consistent volume.
    """
    rms = np.sqrt(np.mean(audio_data**2))
    if rms == 0:
        return audio_data # Avoid division by zero
    gain = target_rms / rms
    normalized_data = audio_data * gain
    # Clip to prevent values exceeding 1.0 (for float audio) or int max (for int audio)
    # assuming audio_data is already float within [-1, 1] range.
    normalized_data = np.clip(normalized_data, -1.0, 1.0)
    return normalized_data

def speak_processed(text,
                    espeak_ng_args="",
                    sample_rate=22050,  # eSpeak-ng default output rate
                    lowpass_cutoff_hz=4000, # Adjust this: lower makes it more muffled, higher more harsh
                    normalize_target_rms=0.15, # Adjust this: higher means louder
                    volume_multiplier=1.0): # Further adjust overall playback volume
    """
    Generates speech using espeak-ng, applies processing, and plays it.

    Args:
        text (str): The text to speak.
        espeak_ng_args (str): Additional arguments to pass to espeak-ng
                              (e.g., "-v en-us", "-s 150").
        sample_rate (int): The expected sample rate from espeak-ng.
        lowpass_cutoff_hz (int): Cutoff frequency for the low-pass filter.
                                 Lower values remove more high frequencies.
        normalize_target_rms (float): Target RMS level for normalization.
                                      Should be between 0.0 and 1.0.
        volume_multiplier (float): Multiplies the final audio amplitude.
    """
    print(f"Generating speech for: \"{text}\"")

    # 1. Run espeak-ng and capture raw audio to stdout
    command = ["espeak-ng", f'"{text}"', "--stdout"]
    if espeak_ng_args:
        command.extend(espeak_ng_args.split())

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            check=True,
            shell=False # Set to False to prevent shell injection, handle args correctly
        )
        raw_audio_data = process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running espeak-ng: {e}", file=sys.stderr)
        print(f"espeak-ng stdout: {e.stdout.decode()}", file=sys.stderr)
        print(f"espeak-ng stderr: {e.stderr.decode()}", file=sys.stderr)
        return
    except FileNotFoundError:
        print("Error: espeak-ng not found. Please ensure it's installed and in your PATH.", file=sys.stderr)
        return

    if not raw_audio_data:
        print("espeak-ng returned no audio data.", file=sys.stderr)
        return

    # 2. Convert raw bytes to numpy array (signed 16-bit integers)
    audio_np = np.frombuffer(raw_audio_data, dtype=np.int16)

    # Convert to float for processing [-1.0, 1.0]
    audio_float = audio_np.astype(np.float32) / 32768.0

    # 3. Apply Audio Processing
    processed_audio = normalize_audio(audio_float, target_rms=normalize_target_rms)
    processed_audio = apply_lowpass_filter(processed_audio, lowpass_cutoff_hz, sample_rate)

    # Apply final volume multiplier
    processed_audio *= volume_multiplier
    processed_audio = np.clip(processed_audio, -1.0, 1.0) # Ensure no clipping after volume adjustment

    # Convert back to int16 for PyAudio playback
    final_audio_int16 = (processed_audio * 32767).astype(np.int16)

    # 4. Play the processed audio using PyAudio
    p = get_pyaudio_instance()
    stream = p.open(format=p.get_format_from_width(2), # 2 bytes = 16-bit
                    channels=1,
                    rate=sample_rate,
                    output=True)

    try:
        # PyAudio expects bytes
        stream.write(final_audio_int16.tobytes())
    except Exception as e:
        print(f"Error playing audio: {e}", file=sys.stderr)
    finally:
        stream.stop_stream()
        stream.close()
        # p.terminate() # Do not terminate here, as the instance is global and might be reused
                      # Terminate only when the script is fully done, e.g., in main() or atexit

def main():
    # Ensure all dependencies are met before proceeding
    if not ensure_dependencies():
        print("Exiting due to unmet dependencies.")
        sys.exit(1)

    # Now that PyAudio is confirmed, import it here explicitly if needed
    # (though get_pyaudio_instance handles it implicitly)
    # import pyaudio # Re-import after ensure_dependencies might have installed it

    # Example Usage:
    # Basic usage with default processing
    print("\n--- Playing with default processing ---")
    speak_processed("Hello, this is a test of the enhanced espeak-ng audio.", espeak_ng_args="-v en-us")

    # Adjusting low-pass filter for more or less muffled sound
    print("\n--- Adjusting low-pass filter (more muffled) ---")
    speak_processed("This sounds a bit more muffled with a lower cutoff.",
                    espeak_ng_args="-v en-us",
                    lowpass_cutoff_hz=2500) # Lower cutoff

    print("\n--- Adjusting low-pass filter (less muffled, potentially harsher) ---")
    speak_processed("This might be a bit clearer, with a higher cutoff.",
                    espeak_ng_args="-v en-us",
                    lowpass_cutoff_hz=5000) # Higher cutoff

    # Adjusting target RMS for loudness
    print("\n--- Adjusting target RMS (louder) ---")
    speak_processed("Now with a slightly louder normalization target.",
                    espeak_ng_args="-v en-us",
                    normalize_target_rms=0.25)

    print("\n--- Using a different espeak-ng voice ---")
    speak_processed("Using a different voice and processing.",
                    espeak_ng_args="-v en-us+f3")

    # You can also pass more complex arguments directly to espeak-ng
    # For example, speaking speed (-s), pitch (-p), or amplitude (-a)
    print("\n--- Faster speech with custom amplitude ---")
    speak_processed("This is faster speech.",
                    espeak_ng_args="-v en-us -s 180 -a 80") # Note: -a in espeak-ng itself is 0-200.
                                                           # Our wrapper normalizes after espeak-ng's -a.

    # Example of running from command line (requires argparse to parse arguments)
    # For simple testing:
    if len(sys.argv) > 1:
        text_from_cli = " ".join(sys.argv[1:])
        print(f"\n--- Speaking from command line: {text_from_cli} ---")
        speak_processed(text_from_cli, espeak_ng_args="-v en-us")

    # Terminate PyAudio instance when done
    global p_audio_instance
    if p_audio_instance:
        p_audio_instance.terminate()
        p_audio_instance = None # Clear it

if __name__ == "__main__":
    main()
