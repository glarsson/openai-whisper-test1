import os
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
import sys
import numpy as np
sys.path.append('C:\SOURCE\GERRY\openai-whisper-test1')

# set the index to zero so we can start at the beginning of the source file every time
file_index = 0

# Create a custom logger
logger = logging.getLogger(__name__)

# Set the logger's level to INFO
logger.setLevel(logging.WARNING)
logger.setLevel(logging.WARNING)

# Create handlers
c_handler = logging.FileHandler('output/log.txt')
c_handler.setLevel(logging.WARNING)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(message)s')
c_handler.setFormatter(c_format)

# Add handlers to the logger
logger.addHandler(c_handler)

# Initialize CUDA primarily, fall back to CPU if CUDA is not available (disaster for performance)
if torch.cuda.is_available():
    compute_device = "cuda"
else:
    compute_device = "cpu"

# load the processor
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

# load the model
speecht5model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(compute_device)

# load the vocoder, that is the voice encoder
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(compute_device)

# we load this dataset to get the speaker embeddings
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

# speaker ids from the embeddings dataset
speakers = {
    'awb': 0,  # Scottish male
    'bdl': 1138,  # US male
    'clb': 2271,  # US female
    'jmk': 3403,  # Canadian male
    'ksp': 4535,  # Indian male
    'rms': 5667,  # US male
    'slt': 6799  # US female
}

# the 'speaker' variable is the specific voice you want to use from the list above, "awb", "bdl", "clb", etc.
speaker = "clb"

def remove_non_unicode(text):
    return ''.join(char for char in text if unicodedata.category(char) != 'C')

def remove_non_ascii(text):
    return ''.join(char for char in text if ord(char) < 128)

# convert readable ASCII text into voice audio

# Open the file and read the lines into a list
#with open('output/voice.txt', 'r') as file:
#    lines = file.readlines()
def text_to_speech(input_string):
    while True:
        # read file_index from lines into a variable called input_string
        #input_string = lines[file_index]

        inputs = processor(text=input_string, return_tensors="pt").to(compute_device)
        # get the speaker index
        speaker_index = speakers[speaker]
        # load xvector containing speaker's voice characteristics from a dataset
        speaker_embeddings = torch.tensor(embeddings_dataset[speaker_index]["xvector"]).unsqueeze(0).to(compute_device)
        # generate speech with the models
        speech = speecht5model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        # Convert the tensor to numpy array
        speech_np = speech.cpu().numpy()
        
        # Create a PyAudio object
        p = pyaudio.PyAudio()
        
        # Open a stream
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=16000,
                        output=True)
        
        # Write the numpy array data into the stream
        stream.write(speech_np.astype(np.float32).tobytes())
        
        # Close the stream and the PyAudio object
        stream.stop_stream()
        stream.close()
        p.terminate()        
        
        '''
        output_filename = f"output/from_text_to_speech_output{file_index}.wav"
        # save the generated speech to a file with 16KHz sampling rate
        sf.write(output_filename, speech.cpu().numpy(), samplerate=16000)
        wf = wave.open(output_filename, 'rb')
        # Create a PyAudio object
        p = pyaudio.PyAudio()
        # Open a stream
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        # Read data from the file and play it
        data = wf.readframes(1024)
        while len(data) > 0:
            stream.write(data)
            data = wf.readframes(1024)
        # Close the stream and the PyAudio object
        stream.stop_stream()
        stream.close()
        p.terminate()
        '''
        # update the file_index
        #file_index += 1
