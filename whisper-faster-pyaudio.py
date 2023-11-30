import os
from faster_whisper import WhisperModel
import atexit
import pyaudio
import numpy as np
import time
import threading
from scipy.io.wavfile import write
from colorama import Fore

# clear the output folder before every run
for file in os.listdir("output"):
    os.remove(os.path.join("output", file))

# Set the sample rate and duration
sample_rate = 48000  # Sample rate in Hz
segment_duration = 3.0  # Duration in seconds

# Initialize the audio/buffer index variable
audio_array_index = 0
audio_buffer_index = 0

# Initialize the last_transcribed_index as 0
last_transcribed_index = 0

# Initialize the audio_buffer as an empty list
audio_buffer = []

# Initialize the audio_indices as an empty list
audio_indices = [0]

# loudness scaling factor - more is louder
audio_scaling_factor = 3.0  # Adjust this as needed

# whisper-faster inits
model_size = "medium"
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

# PyAudio setup
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=int(sample_rate * segment_duration))

def record_audio():
    global audio_buffer
    global audio_indices
    global audio_array_index

    while True:
        # Record audio
        #print(f"Recording audio for {segment_duration} seconds...")
        audio = np.frombuffer(stream.read(int(sample_rate * segment_duration), exception_on_overflow=False), dtype=np.int16)

        # Convert the audio data to a floating-point format
        audio_float = audio.astype(np.float32)

        # Scale each sample individually to prevent distortion
        audio_float_scaled = audio_float * audio_scaling_factor

        # Convert the scaled audio data back to 16-bit PCM format
        audio_pcm = np.int16(audio_float_scaled)

        # Ensure audio_pcm is a 1D array
        audio_pcm = audio_pcm.reshape(-1)

        # Append the audio data to the buffer
        audio_buffer.extend(audio_pcm)

        # Update the current length of the buffer in audio_indices
        audio_indices.append(len(audio_buffer))

        #print("Recording saved to array. Array index is now at %s" % (audio_array_index))

        # Increment the audio_array_index
        audio_array_index += 1

        # Add a short sleep to avoid overloading the buffer
        time.sleep(0.1)

def convert_array_to_wave():
    global audio_buffer
    global audio_indices
    global last_transcribed_index

    while True:
        #print("Starting to convert buffer to wave...")
        global audio_buffer
        global audio_indices
        global last_transcribed_index

        # Wait until there is new audio data in the buffer
        while len(audio_indices) <= last_transcribed_index + 1:
            time.sleep(0.1)  # Sleep for a short time to avoid busy waiting

        #print("Found audio in buffer. Converting to wave...")

        # Extract the new audio data from the audio_buffer
        start_index = audio_indices[last_transcribed_index]
        end_index = audio_indices[last_transcribed_index + 1]
        audio_data = audio_buffer[start_index:end_index]

        #print("Extracted audio data from buffer. Writing to file...")

        # Create a temporary file name using the current index
        temp_file_name = f"output/output_{last_transcribed_index}.wav"

        # Write the audio data to the temporary file
        write(temp_file_name, sample_rate, np.array(audio_data, dtype=np.int16))

        #print("Wrote audio data to file.")

        last_transcribed_index += 1

def transcribe_audio():
    transcriber_file_index = 0
    while True:
        #print("Starting a transcribe audio run at internal index %s" % (transcriber_file_index))

        # Create a temporary "pointer" using the current index
        temp_file_name = f"output/output_{transcriber_file_index}.wav"

        # check that temp_file_name exists on the file system, otherwise wait for it to be written
        while not os.path.exists(temp_file_name) or os.path.getsize(temp_file_name) < 10240:
            time.sleep(0.5)

        #print(f"Found audio file {temp_file_name}. Transcribing...")

        # Transcribe the audio
        segments, info = model.transcribe(temp_file_name, beam_size=5)

        #print(f"segments loaded model and file {temp_file_name}. Transcribing")

        # Initialize an empty list to hold the transcriptions
        transcriptions = []

        # Add the language detection result to the transcriptions
        # transcriptions.append("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        # Add the segment transcriptions to the list
        for segment in segments:
            transcriptions.append("%s" % (segment.text))

        # Join the words into a single string
        transcription_string = ' '.join(transcriptions)

        #print(f"Transcription completed for internal index {transcriber_file_index}.")

        # increment local index
        transcriber_file_index += 1
        
        # Print the transcription string
        print(Fore.GREEN + f"{transcription_string}" + Fore.RESET)


























# Create a thread for the record_audio function
record_thread = threading.Thread(target=record_audio)

# Create a thread for the convert_array_to_wave function
convert_thread = threading.Thread(target=convert_array_to_wave)

# Create a thread for the transcribe function
transcribe_thread = threading.Thread(target=transcribe_audio)

# Start the threads
record_thread.start()
convert_thread.start()
transcribe_thread.start()


# Stop the PyAudio stream when the program exits
atexit.register(lambda: stream.stop_stream())
atexit.register(lambda: stream.close())
atexit.register(lambda: p.terminate())
