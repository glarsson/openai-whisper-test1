import os
import queue
import logging
import time
import wave
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import pyaudio
import unicodedata
from faster_whisper import WhisperModel
from colorama import Fore
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import random
import string
import soundfile as sf
from datetime import datetime

# Initialize the index for the speech_to_text transcriber
speech_to_text_transcriber_file_index = 0


# clear the output folder before every run
for file in os.listdir("output"):
    os.remove(os.path.join("output", file))

# Create a custom logger
logger = logging.getLogger(__name__)
# Set the logger's level to INFO
logger.setLevel(logging.WARNING)
logger.setLevel(logging.WARNING)# Create handlers
c_handler = logging.FileHandler('output/log.txt')
c_handler.setLevel(logging.WARNING)
# Create formatters and add it to handlers
c_format = logging.Formatter('%(message)s')
c_handler.setFormatter(c_format)
# Add handlers to the logger
logger.addHandler(c_handler)

# initialize CUDA primarily, fall back to CPU if CUDA is not available (disaster for performance)

if torch.cuda.is_available():
    compute_device = "cuda"
else: compute_device = "cpu"

# INITIALIZATION FOR WHISPER (SPEECH TO TEXT)
# Start the timer
start_time = time.time()
# whisper-faster inits
# model_size can be "small" or "large" - obvious tradeoffs
model_size = "small"
model = WhisperModel(model_size, device=compute_device, compute_type="int8_float16")
# beam_size:
# Larger = more accurate, but slower performance (it considers more possibilities and thus LLM-style finding the best path)
# Smaller = less accurate, but faster performance
# 5 is a good balance between speed and accuracy for the small model, but 3 seems to do the trick for now
whisper_beam_size = 3
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
# Log the elapsed time
# Get the current time of day
current_time = datetime.now().strftime("%H:%M:%S.%f")
logger.info(current_time + f" Loading Whisper-Faster model(s) took {elapsed_time} seconds ({elapsed_time * 1000} milliseconds)")

#### START OF FUNCTIONS ####

def remove_non_unicode(text):
    return ''.join(char for char in text if unicodedata.category(char)[0] != 'C')

def remove_non_ascii(text):
    return ''.join(char for char in text if ord(char) < 128)

# convert voice audio into readable ASCII text
def speech_to_text():
    current_time = datetime.now().strftime("%H:%M:%S.%f")
    logger.info(current_time + " Whisper-Faster model loaded")
    audiofile_to_transcribe = ""
    global speech_to_text_transcriber_file_index

    while True:
        global speech_to_text_transcriber_file_index

        current_time = datetime.now().strftime("%H:%M:%S.%f")
        audiofile_to_transcribe = (f"c:/source/gerry/openai-whisper-test1/output/audio_input_convert_array_to_wave-output_{speech_to_text_transcriber_file_index}.wav")
        logger.info(f"{current_time} [speech_to_text] looking/waiting for input audio: {audiofile_to_transcribe}")

        #print("%s [speech_to_text] Looking for input audio: with index %s" % (current_time, text_to_speech_transcriber_file_index))
        #print("looking for output/from_speech_to_text_input_%s.wav" % (text_to_speech_transcriber_file_index))

        # Check if the file exists and is not empty
        while not os.path.exists(audiofile_to_transcribe) or os.path.getsize(audiofile_to_transcribe) == 0:
            time.sleep(1.0)  # check every 200ms
            #print(f"file {audiofile_to_transcribe} not found, waiting 1 second.")
        else:
            current_time = datetime.now().strftime("%H:%M:%S.%f")
            logger.info(f"{current_time} [speech_to_text] Found audio: {audiofile_to_transcribe}")
            # Start the timer
            start_time = time.time()

            # Transcribe the audio
            segments, info = model.transcribe(audiofile_to_transcribe, beam_size=whisper_beam_size)

            # Stop the timer
            end_time = time.time()

            # Calculate the elapsed time
            elapsed_time = end_time - start_time

            # Log the elapsed time
            current_time = datetime.now().strftime("%H:%M:%S.%f")
            logger.info(f"{current_time} [speech_to_text] Transcription took {elapsed_time} seconds ({elapsed_time * 1000} milliseconds)")

            # Increment the file index since we've transcibed already
            speech_to_text_transcriber_file_index += 1

            # Initialize an empty list to hold the transcriptions
            transcriptions = []

            # Add the segment transcriptions to the list

            for segment in segments:
                start_time = segment.start
                end_time = segment.end
                text = segment.text
                # Remove non-ascii characters, it hallucinates unicode characters when there is static/music/noise etc, 
                # it can also speak and Japanese, Chinese etc so we'll remove those because we don't need them (yet)
                text = remove_non_ascii(text)
                text = remove_non_unicode(text)
                # append output to file
                with open('output/voice.txt', 'a', encoding='utf-8') as f:
                    #current_time = datetime.now().strftime("%H:%M:%S.%f")
                    #f.write(current_time + " " + "[voice]" + transcription_string + "\n")
                    f.write(f"{text}")
                transcriptions.append((start_time, end_time, text))

            transcription_string = ' '.join(text for _, _, text in transcriptions)
            current_time = datetime.now().strftime("%H:%M:%S.%f")
            logger.info(f"{current_time} [speech_to_text] segments have been joined.")

            # It has a habit of hallucinating words like "you" and "thank you", etc. So we'll (do our best to) remove those
            unwanted_words = ["you", "Thank you.", "Thanks for watching!"]
            if not any(word in transcription_string and transcription_string.count(word) == 1 for word in unwanted_words):
                current_time = datetime.now().strftime("%H:%M:%S.%f")
                logger.info(f"{current_time} [speech_to_text] {transcription_string}")

                
                
                # this is the output to the console, would be cool to have certain words highlighted in certain colors
                


                print(Fore.YELLOW + f"[speech_to_text]{transcription_string}" + Fore.RESET)
                


                current_time = datetime.now().strftime("%H:%M:%S.%f")
                logger.info(f"{current_time} [speech_to_text] Appended transcripton to voice.txt - sending to text_to_speech")

                # globals.global_tts_input_string = transcription_string
                # increment local index
