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

# Initialize the indicies for the text_to_speech and speech_to_text transcribers, they may not run at the same speed so we need to keep track of them separately
text_to_speech_transcriber_file_index = 0
speech_to_text_transcriber_file_index = 0


# clear the output folder before every run
for file in os.listdir("output"):
    os.remove(os.path.join("output", file))

# Create a custom logger
logger = logging.getLogger(__name__)
# Set the logger's level to INFO
logger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)# Create handlers
c_handler = logging.FileHandler('output/log.txt')
c_handler.setLevel(logging.INFO)
# Create formatters and add it to handlers
c_format = logging.Formatter('%(message)s')
c_handler.setFormatter(c_format)
# Add handlers to the logger
logger.addHandler(c_handler)


# Initialize a global variable to hold the input string for text to speech, we're setting it to THREADPAUSE to begin with
# and then after each successful text_to_speech pushes content to this, we'll replace it with "THREADPAUSE" as soon as it gets processed so text_to_speech knows to wait for a new one
global_tts_input_string = "THREADPAUSE"

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

### INITIALIZATION FOR TTS (TEXT TO SPEECH) ###
# Start the timer
start_time = time.time()
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
    'awb': 0,     # Scottish male
    'bdl': 1138,  # US male
    'clb': 2271,  # US female
    'jmk': 3403,  # Canadian male
    'ksp': 4535,  # Indian male
    'rms': 5667,  # US male
    'slt': 6799   # US female
}
# the 'speaker' variable is the specific voice you want to use from the list above, "awb", "bdl", "clb", etc.
speaker = "clb"
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
# Log the elapsed time
# Get the current time of day
current_time = datetime.now().strftime("%H:%M:%S.%f")
logger.info(current_time + f" Loading TTS model(s) took {elapsed_time} seconds ({elapsed_time * 1000} milliseconds)")

#### START OF FUNCTIONS ####

def remove_non_unicode(text):
    return ''.join(char for char in text if unicodedata.category(char)[0] != 'C')

def remove_non_ascii(text):
    return ''.join(char for char in text if ord(char) < 128)

# convert readable ASCII text into voice audio
def text_to_speech():
    current_time = datetime.now().strftime("%H:%M:%S.%f")
    logger.info(f"current_time + " " + [text_to_speech] " + "TTS model(s) loaded")
    global global_tts_input_string
    global text_to_speech_transcriber_file_index

    while True:
        global global_tts_input_string
        # Check if the variable has been set to THREADPAUSE, if so, wait for a new input string
        while global_tts_input_string == "THREADPAUSE" or global_tts_input_string is None:
            time.sleep(0.2)  # check every 200ms
            #print(f"global_tts_input_string is THREADPAUSE, waiting 1 second...")
        else:
            inputs = processor(text=global_tts_input_string, return_tensors="pt").to(compute_device)
            # get the speaker index
            speaker_index = speakers[speaker]
            # load xvector containing speaker's voice characteristics from a dataset
            speaker_embeddings = torch.tensor(embeddings_dataset[speaker_index]["xvector"]).unsqueeze(0).to(compute_device)
            # generate speech with the models
            speech = speecht5model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
            output_filename = f"output/from_text_to_speech_output{last_speech_transcribed_index}.mp3"
            # save the generated speech to a file with 16KHz sampling rate
            sf.write(output_filename, speech.gpu().numpy(), samplerate=16000)
            # let's try to just output the audio directly here without any fancy shit
            #debug
            print("set global_tts_input_string back to THREADPAUSE")
            # set global_tts_input_string back to THREADPAUSE
            global_tts_input_string = "THREADPAUSE"    
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

# convert voice audio into readable ASCII text
def speech_to_text():
    current_time = datetime.now().strftime("%H:%M:%S.%f")
    logger.info(current_time + " Whisper-Faster model loaded")
    global global_tts_input_string
    global last_speech_transcribed_index
    audiofile_to_transcribe = ""
    global text_to_speech_transcriber_file_index
    global speech_to_text_transcriber_file_index

    while True:
        global text_to_speech_transcriber_file_index
        global speech_to_text_transcriber_file_index

        current_time = datetime.now().strftime("%H:%M:%S.%f")
        logger.info(f"{current_time} [speech_to_text] looking/waiting for input audio: {audiofile_to_transcribe}")
        print("%s [speech_to_text] Looking for input audio: with index %s" % (current_time, text_to_speech_transcriber_file_index))
        print("looking for output/from_speech_to_text_input_%s.wav" % (text_to_speech_transcriber_file_index))

        audiofile_to_transcribe = (f"c:/source/gerry/openai-whisper-test1/output/audio_input_convert_array_to_wave-output_{speech_to_text_transcriber_file_index}.wav")

        # Check if the file exists and is not empty
        while not os.path.exists(audiofile_to_transcribe) or os.path.getsize(audiofile_to_transcribe) == 0:
            time.sleep(1.0)  # check every 200ms
            #print(f"file {audiofile_to_transcribe} not found, waiting 1 second...")
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

            # Initialize an empty list to hold the transcriptions
            transcriptions = []

            # Add the segment transcriptions to the list

            for segment in segments:
                start_time = segment.start
                end_time = segment.end
                text = segment.text
                transcriptions.append((start_time, end_time, text))

            transcription_string = ' '.join(text for _, _, text in transcriptions)
            current_time = datetime.now().strftime("%H:%M:%S.%f")
            logger.info(f"{current_time} [speech_to_text] segments have been joined.")

            # Remove non-ascii characters, it hallucinates unicode characters when there is static/music/noise etc, 
            # it can also speak and Japanese, Chinese etc so we'll remove those because we don't need them (yet)
            transcription_string = remove_non_ascii(transcription_string)

            # It has a habit of hallucinating words like "you" and "thank you", etc. So we'll (do our best to) remove those
            unwanted_words = ["you", "Thank you.", "Thanks for watching!"]
            if not any(word in transcription_string and transcription_string.count(word) == 1 for word in unwanted_words):
                current_time = datetime.now().strftime("%H:%M:%S.%f")
                logger.info(f"{current_time} [speech_to_text] {transcription_string}")

                
                
                # this is the output to the console, would be cool to have certain words highlighted in certain colors
                


                print(Fore.YELLOW + f"{current_time} [speech_to_text] {transcription_string}" + Fore.RESET)
                
                # append output to file
                with open('output/voice.txt', 'a', encoding='utf-8') as f:
                    #current_time = datetime.now().strftime("%H:%M:%S.%f")
                    #f.write(current_time + " " + "[voice]" + transcription_string + "\n")
                    f.write(f"{transcription_string}\n")

                current_time = datetime.now().strftime("%H:%M:%S.%f")
                logger.info(f"{current_time} [speech_to_text] Appended transcripton to voice.txt - sending to text_to_speech")

                global_tts_input_string = transcription_string
                # increment local index
                speech_to_text_transcriber_file_index += 1