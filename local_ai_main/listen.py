import os
import threading
import time

from operations.audio_input import record_audio
from operations.audio_input import convert_array_to_wave
from operations.audio_transcription import speech_to_text
from operations.audio_transcription import text_to_speech
#from operations.audio_output import text_to_speech

# Add a variable to keep track of the last processed index
# This is for a technique to feed one second of the previous audio to the next transcription to avoid
# missing words that might get cut off, let's see if it works!
last_processed_index = 0

### THREADS ###

# Create a thread for the speech to text function
speech_to_text_thread = threading.Thread(target=speech_to_text)
speech_to_text_thread.start()

# Create a thread for the text to speech function
text_to_speech_thread = threading.Thread(target=text_to_speech)
text_to_speech_thread.start()

# Create a thread for the record_audio function
record_thread = threading.Thread(target=record_audio)
record_thread.start()

# Create a thread for the convert_array_to_wave function
convert_thread = threading.Thread(target=convert_array_to_wave)
convert_thread.start()
