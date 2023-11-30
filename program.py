import base64
from openai import OpenAI
from colorama import Fore
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import keyboard
import wave
from faster_whisper import WhisperModel
import time
import threading


# Read API key from a file
with open('secret_apikey.txt', 'r') as file:
    api_key = file.read().strip()

client = OpenAI(api_key=api_key)


## let's have a look at recording some audio for whisper
# List all available sound devices
print(sd.query_devices())

# Set the device to the default device
device_info = sd.query_devices(None, 'input')
device = device_info['name']

# Set the sample rate and duration
sample_rate = 44100  # Sample rate in Hz
duration = 5.0  # Duration in seconds

'''
# Record audio
print(f"Recording audio from {device} for {duration} seconds...")
audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=2)
sd.wait()  # Wait for the recording to finish

# Convert the audio data to a PCM format
audio_pcm = np.int16(audio / np.max(np.abs(audio)) * 32767)

# Save the audio to a wave file
write('output.wav', sample_rate, audio_pcm)
print("Recording saved to output.wav")

# Open the wave file
with wave.open('output.wav', 'rb') as wav_file:
    # Get the number of frames
    num_frames = wav_file.getnframes()

# Check if the file contains sound
if num_frames > 0:
    print("The file contains sound.")
else:
    print("The file does not contain sound.")
'''

# faster-whisper stuff

# model_size = "large-v3"
model_size = "large-v2"

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8

# we shall do this, it says ~3GB VRAM required for this one
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

'''
segments, info = model.transcribe("output.wav", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

segments, _ = model.transcribe("output.wav")
segments = list(segments)  # The transcription will actually run here.
segments, _ = model.transcribe("output.wav", word_timestamps=True)

for segment in segments:
    for word in segment.words:
        print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
'''

# Set the duration of each audio segment
segment_duration = 1.0  # Duration in seconds

def record_and_transcribe():
    while True:
        # Record audio
        audio = sd.rec(int(sample_rate * segment_duration), samplerate=sample_rate, channels=2)
        sd.wait()  # Wait for the recording to finish

        # Convert the audio data to a PCM format
        audio_pcm = np.int16(audio / np.max(np.abs(audio)) * 32767)

        # Save the audio to a temporary wave file
        write('temp.wav', sample_rate, audio_pcm)

        # Transcribe the audio
        segments, _ = model.transcribe('temp.wav')
        segments = list(segments)  # The transcription will actually run here.

        for segment in segments:
            if segment.words is not None:  # Add this check
                for word in segment.words:
                  print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))

# Start the recording and transcription in a separate thread
threading.Thread(target=record_and_transcribe).start()






















'''
# Our first AI 'assistant' role and its speciality
sme1_specialization = "UI/UX programmer"
openai_sme1 = f"As an expert {sme1_specialization}, my take is:"

# Our second AI 'assistant' role and its speciality
sme2_specialization = "Backend programmer"
openai_sme2 = f"As an expert {sme2_specialization}, my take is:"

# The base premise of what we are trying to do
base_premise = f"""
The following will be a user started, but AI led conversation - multi input/output from different assistants to reach a final conclusion.
It begins with the user asking a question which we will pipe to different openai assistants.

Our two experts are {sme1_specialization} and {sme2_specialization}
"""
print(base_premise)

# Get user input
# user_input = input("Please enter your question: ")

with open('game_rules.txt', 'r') as f:
    game_rules = f.read()

system_input = "The goal is to create a web-based game called Mulle." + game_rules

# we're gonna start to do some whisper stuff later, it seems troublesome to get audio recorded in python and windows for some reason.
# print(f"{Fore.WHITE}Whisper things you said: {user_input}{Fore.RESET}\n")

# first sme response
sme1_response = client.chat.completions.create(
  model="gpt-3.5-turbo-16k-0613",
  max_tokens=4096,
  messages=[
    {"role": "system", "content": system_input,
     "role": "assistant", "content": "What is the best way to achieve this? maybe remix? another framework?"
    }
    ]
)

sme1_response_text = sme1_response.choices[0].message.content
print(f"{Fore.CYAN}The {sme1_specialization} says: {sme1_response_text}{Fore.RESET}\n")

# second  sme response
sme2_response = client.chat.completions.create(
  model="gpt-3.5-turbo-16k-0613",
  max_tokens=4096,
  messages=[
    {"role": "system", "content": system_input,
     "role": "assistant", "content": "What is the best way to achieve this? maybe c#? dotnet?"
    }
    ]
)

sme2_response_text = sme2_response.choices[0].message.content
print(f"{Fore.MAGENTA}The {sme2_specialization} says: {sme2_response_text}{Fore.RESET}\n")

# initialize summarizer
init_summarizer = "please come up with the best way to write this card game, maybe some example code"
openai_summarizer = init_summarizer +  openai_sme1 + ": " + sme1_response_text + ". " + openai_sme2 + ": " + sme2_response_text

# summarizer response
summarizer_response = client.chat.completions.create(
  model="gpt-3.5-turbo-16k-0613",
  max_tokens=4096,
  messages=[
    {"role": "assistant", "content": openai_summarizer }
  ]
)

summarizer_response_text = summarizer_response.choices[0].message.content
print(f"{Fore.YELLOW}The summarizer says: {summarizer_response_text}{Fore.RESET}\n")

# initialize conversation
# init_conversation = "please summarize the following two statements as condensed as possible:"
conversation_response = client.chat.completions.create(
  model="gpt-3.5-turbo-16k-0613",
  max_tokens=2048,
  messages=[
    {"role": "system", "content": "Please discuss amongst yourselves, the answers provided:" + summarizer_response_text},
    {"role": "assistant", "content": sme1_response_text},
    {"role": "assistant", "content": sme2_response_text}
  ]
)

conversation_response_text = conversation_response.choices[0].message.content
print(f"{Fore.RED}The conversational response is: {conversation_response_text}{Fore.RESET}\n")


# the final answer assistant is pretty useless - the summarizer is doing a better job so it's kind of redundant right now.

#final_answer_init = "Your job is now to consider the expert opinions and the summarizer's condensed version of them and come up with a final answer to the question."
#final_answer_prompt = f"The {sme1_specialization} says: {sme1_response_text} The {sme2_specialization} says: {sme2_response_text} The summarizer says: {summarizer_response_text}"
# final response
#final_response = client.chat.completions.create(
#  model="gpt-3.5-turbo-16k-0613",
#  messages=[
#    {"role": "system", "content": base_premise },
#    {"role": "assistant", "content": final_answer_init + final_answer_prompt }
#  ]
#)

#final_response_text = final_response.choices[0].message.content
#print(f"{Fore.GREEN}The final answer is: {final_response_text}{Fore.RESET}\n")
'''