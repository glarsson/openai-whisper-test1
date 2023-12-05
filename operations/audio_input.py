import time
import numpy as np
import pyaudio
from scipy.io.wavfile import write
import atexit
import audioop


# Set the sample rate and duration
sample_rate = 48000  # Sample rate in Hz

# loudness scaling factor - more is louder
audio_scaling_factor = 4.0  # Adjust this as needed

# Setting this really has a big impact on the accuuracy of the transcription but also the speed of output - it doesn't stream.
# 3 seconds is acceptable I think for normal conversational rythm, 
# setting this for 4 got me a lot better accuracy with fast speaking
segment_duration = 4  # Duration in seconds

# Initialize the audio_buffer as an empty list
audio_buffer = []

# Initialize the audio_indices as an empty list
audio_indices = [0]

# Initialize the audio_array_index as 0
audio_array_index = 0

global previous_audio
previous_audio = []

# Initialize the (speech to text) last_transcribed_index as 0
last_transcribed_index = 0

# pyaudio setup
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
        # Maybe this step takes some time, should check later if its needed
        audio_pcm = audio_pcm.reshape(-1)

        # Append the audio data to the buffer
        audio_buffer.extend(audio_pcm)

        # Update the current length of the buffer in audio_indices
        audio_indices.append(len(audio_buffer))

        #print("Recording saved to array. Array index is now at %s" % (audio_array_index))

        # Increment the audio_array_index
        audio_array_index += 1

        # Add a short sleep to avoid overloading the buffer
        # Why? I don't understand why I put this here, copilot suggested it. It's got something to do with poor mans threading I guess.
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
        
        # Prepend previous_audio to audio_data if last_transcribed_index > 0
        if last_transcribed_index > 0:
            audio_data = previous_audio + audio_data

        #print("Extracted audio data from buffer. Writing to file...")

        # Create a temporary file name using the current index
        temp_file_name = f"output/audio_input_convert_array_to_wave-output_{last_transcribed_index}.wav"

        # Write the audio data to the temporary file
        write(temp_file_name, sample_rate, np.array(audio_data, dtype=np.int16))

        #print("Wrote audio data to file.")

        last_transcribed_index += 1

# Stop the PyAudio stream when the program exits
atexit.register(lambda: stream.stop_stream())
atexit.register(lambda: stream.close())
atexit.register(lambda: p.terminate())