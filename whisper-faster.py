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
model_size = "medium"
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

## let's have a look at recording some audio for whisper
# List all available sound devices
print(sd.query_devices())

# Set the device to the default device
device_info = sd.query_devices(None, 'input')
device = device_info['name']

# Set the sample rate and duration
sample_rate = 48000  # Sample rate in Hz

# Set the duration of each audio segment
segment_duration = 3.0  # Duration in seconds

# Initialize the audio/buffer index variable
audio_array_index = 0
audio_buffer_index = 0

# Initialize the last_transcribed_index as 0
last_transcribed_index = 0

# Initialize the audio_buffer as an empty numpy array
audio_buffer = np.array([])

# Initialize the audio_indices as an empty list
audio_indices = [0]

audio_scaling_factor = 0.1  # Adjust this as needed

def record_audio():
    global audio_buffer  # Use the global audio_buffer variable
    global audio_indices  # Use the global audio_indices list
    global audio_array_index

    while True:
        # Record audio
        print(f"Recording audio from {device} for {segment_duration} seconds...")
        audio = sd.rec(int(sample_rate * segment_duration), samplerate=sample_rate, channels=1, gain=0.5)
        sd.wait()  # Wait for the recording to finish

        # Convert the audio data to a 16-bit PCM format with reduced scaling
        audio_pcm = np.int16(np.mean(audio, axis=1) * (32767 * audio_scaling_factor))

        # Append the audio data to the buffer
        audio_buffer = np.concatenate((audio_buffer, audio_pcm))

        # Append the current length of the buffer to audio_indices
        audio_indices.append(len(audio_buffer))

        print("Recording saved to array. Array index is now at %s" % (audio_array_index))

        # Increment the audio_array_index
        audio_array_index += 1

def convert_array_to_wave():
    while True:
        print("Starting to convert buffer to wave...")
        global audio_buffer  # Use the global audio_buffer variable
        global audio_indices  # Use the global audio_indices list
        global last_transcribed_index  # Use the global last_transcribed_index variable

        # Wait until there is new audio data in the buffer
        while len(audio_indices) <= last_transcribed_index + 1:
            time.sleep(0.1)  # Sleep for a short time to avoid busy waiting

        print("Found audio in buffer. Converting to wave...")

        # Extract the new audio data from the audio_buffer
        audio_data = audio_buffer[audio_indices[last_transcribed_index]:audio_indices[last_transcribed_index + 1]]

        print("Extracted audio data from buffer. Writing to file...")

        # Create a temporary file name using the current index
        temp_file_name = f"output_{last_transcribed_index}.wav"

        # Write the audio data to the temporary file
        write(temp_file_name, sample_rate, audio_data)

        print("Wrote audio data to file.")

        last_transcribed_index += 1

'''
def transcribe_audio():
    while True:
        # check if file was written
        #print("File written to " + temp_file_name)

        # Transcribe the audio
        segments, info = model.transcribe(temp_file_name, beam_size=5)

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
            transcriptions.append("%s" % (segment.text))

        # Join the words into a single string
        transcription_string = ' '.join(transcriptions)

        # Return the transcription string
        return transcription_string
'''

# Create a thread for the record_audio function
record_thread = threading.Thread(target=record_audio)

# Create a thread for the record_audio function
convert_thread = threading.Thread(target=convert_array_to_wave)


# Create a thread for the transcribe_audio function
#transcribe_audio = threading.Thread(target=transcribe_audio)

# Start the threads
record_thread.start()
convert_thread.start()

