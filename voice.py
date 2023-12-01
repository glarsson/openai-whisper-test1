import os
import threading

from operations.audio_operations import convert_array_to_wave
from operations.audio_transcription import transcribe_audio
from operations.audio_operations import record_audio

# clear the output folder before every run
for file in os.listdir("output"):
    os.remove(os.path.join("output", file))

# Add a variable to keep track of the last processed index
# This is for a technique to feed one second of the previous audio to the next transcription to avoid
# missing words that might get cut off, let's see if it works!
last_processed_index = 0

# Initialize the last_transcribed_index as 0
last_transcribed_index = 0

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