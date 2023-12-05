import sys
import logging
import time
import unicodedata
from colorama import Fore
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datetime import datetime
sys.path.append('C:\SOURCE\GERRY\openai-whisper-test1')
import globals

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

# Initialize the index for the speech_to_text transcriber
text_to_speech_transcriber_file_index = 0

# Initialize a global variable to hold the input string for text to speech, we're setting it to THREADPAUSE to begin with
# and then after each successful text_to_speech pushes content to this, we'll replace it with "THREADPAUSE" as soon as it gets processed so text_to_speech knows to wait for a new one
globals.global_tts_input_string

# initialize CUDA primarily, fall back to CPU if CUDA is not available (disaster for performance)

if torch.cuda.is_available():
    compute_device = "cuda"
else: compute_device = "cpu"

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
print("text to speech started")
# convert readable ASCII text into voice audio
def text_to_speech_conversation():
    current_time = datetime.now().strftime("%H:%M:%S.%f")
    logger.info(f"current_time + " " + [text_to_speech] " + "TTS model(s) loaded")
    globals.global_tts_input_string
    global text_to_speech_transcriber_file_index

    while True:
        globals.global_tts_input_string
        # Check if the variable has been set to THREADPAUSE, if so, wait for a new input string
        while globals.global_tts_input_string == "THREADPAUSE" or globals.global_tts_input_string is None:
            time.sleep(0.1)  # check every 200ms
            #print(f"globals.global_tts_input_string is THREADPAUSE, waiting 1 second...")
        else:
            inputs = processor(text=globals.global_tts_input_string, return_tensors="pt").to(compute_device)
            # get the speaker index
            speaker_index = speakers[speaker]
            # load xvector containing speaker's voice characteristics from a dataset
            speaker_embeddings = torch.tensor(embeddings_dataset[speaker_index]["xvector"]).unsqueeze(0).to(compute_device)
            # generate speech with the models
            speech = speecht5model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
            output_filename = f"output/from_text_to_speech_output{text_to_speech_transcriber_file_index}.wav"
            # save the generated speech to a file with 16KHz sampling rate
            sf.write(output_filename, speech.cpu().numpy(), samplerate=16000)
            # let's try to just output the audio directly here without any fancy shit
            #debug
            #print("set globals.global_tts_input_string back to THREADPAUSE")
            # set globals.global_tts_input_string back to THREADPAUSE
            globals.global_tts_input_string = "THREADPAUSE"    
            '''
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