from faster_whisper import WhisperModel
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import sounddevice as sd
import numpy as np
import wave
import time
import threading
from queue import Queue

# whisper-faster inits
model_size = "large-v2"
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

## let's have a look at recording some audio for whisper
# List all available sound devices
print(sd.query_devices())

# Set the device to the default device
device_info = sd.query_devices(None, 'input')
device = device_info['name']

# Set the sample rate and duration
sample_rate = 44100  # Sample rate in Hz

# Set the duration of each audio segment
segment_duration = 3.0  # Duration in seconds

def record_audio():
    while True:
        # Record audio
        print(f"Recording audio from {device} for {segment_duration} seconds...")
        audio = sd.rec(int(sample_rate * segment_duration), samplerate=sample_rate, channels=2)
        sd.wait()  # Wait for the recording to finish

        # Convert the audio data to a PCM format
        audio_pcm = np.int16(audio / np.max(np.abs(audio)) * 32767)

        # Save the audio to a wave file and overwrite the file if existing
        write('output.wav', sample_rate, audio_pcm)
        print("Recording saved to output.wav")

        # Open the wave file
        with wave.open('output.wav', 'rb') as wav_file:
            # Get the number of frames
            num_frames = wav_file.getnframes()

        # Check if the file contains sound
        if num_frames > 0:
            print("The file contains sound.")
            # call transcribe_audio() and wait for it to finish
            transcriptions = transcribe_audio()
            print("Transcriptions:", transcriptions)
        else:
            print("The file does not contain sound.")

def transcribe_audio():
    segments, info = model.transcribe("output.wav", beam_size=5)

    # Initialize an empty list to hold the transcriptions
    transcriptions = []

    # Add the language detection result to the transcriptions
    # transcriptions.append("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    # Add the segment transcriptions to the list
    for segment in segments:
        transcriptions.append("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    segments, _ = model.transcribe("output.wav")
    segments = list(segments)  # The transcription will actually run here.
    segments, _ = model.transcribe("output.wav", word_timestamps=True)

    # Add the segment transcriptions to the list
    for segment in segments:
        transcriptions.append("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    # Join the words into a single string
    transcription_string = ' '.join(transcriptions)

    # Return the transcription string
    return transcription_string

record_audio()