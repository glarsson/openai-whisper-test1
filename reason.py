
from openai import OpenAI
from colorama import Fore
import numpy as np

# Read API key from a file
with open('secret_apikey.txt', 'r') as file:
    api_key = file.read().strip()

client = OpenAI(api_key=api_key)

# specify the OpenAI model to use:
# https://platform.openai.com/docs/models/gpt-3-5
# 16k token limit on "gpt-3.5-turbo-16k-0613"
# 16k token limit on "gpt-3.5-turbo-1106" (newest release as of dec-1-2023)
gpt_model = "gpt-3.5-turbo-1106"

# Our first AI 'assistant' role and its speciality
sme1_specialization = "film critic"
# The base premise of what we are trying to do
base_premise_vision = f"""
{sme1_specialization}. You are going to analyze the text content after "[vision] " (then ignore that beginning of every line) outputten by another AI and try to interpret what the general vibe 
of the video is through the text. Try to convey through these small snapshots of information on each line what the arist is trying for.
You will then provide a detailed summary of what you think conceptually about the video (that you analyzed in text form), and it's qualities and finally what the message of the video is.
"""
# Our second AI 'assistant' role and its speciality
sme2_specialization = "music critic"
base_premise_voice = f"""
{sme2_specialization}. You are going to analyze the content after "[voice] " (then ignore that beginning of every line) outputten by another AI and try to interpret what the general idea 
of the text is trying to convey through these small snapshots of information on each line. You will then provide a detailed summary of what you think
conceptually about the text, and it's qualities and finally what the message is.
"""

# print(base_premise)

####
####

# First we're gonna have to actually watch the video and grab the content... :D

# So I will do that not in here but separately, later connect them.

####
####

# Get vision content from file
with open('output/vision.txt', 'r', encoding='utf-8') as file:
    vision_analysis = file.read().strip()

# first sme response
sme1_response = client.chat.completions.create(
  model=gpt_model,
  max_tokens=2048,
  messages=[
    {"role": "system", "content": base_premise_vision,
     "role": "user", "content": vision_analysis,
     "role": "assistant", "content": base_premise_vision + sme1_specialization + vision_analysis     
    }
    ]
)

sme1_response_text = sme1_response.choices[0].message.content
print(f"{Fore.CYAN}The {sme1_specialization} says: {sme1_response_text}{Fore.RESET}\n")

# Get voice content from file
with open('output/voice.txt', 'r', encoding='utf-8') as file:
    voice_analysis = file.read().strip()

# second sme response
sme2_response = client.chat.completions.create(
  model=gpt_model,
  max_tokens=2048,
  messages=[
    {"role": "system", "content": base_premise_vision,
     "role": "user", "content": voice_analysis,
     "role": "assistant", "content": base_premise_voice + sme2_specialization + voice_analysis
    }
    ]
)

sme2_response_text = sme2_response.choices[0].message.content
print(f"{Fore.MAGENTA}The {sme2_specialization} says: {sme2_response_text}{Fore.RESET}\n")


'''
# summarizer response
summarizer_response = client.chat.completions.create(
  model=gpt_model,
  max_tokens=1024,
  messages=[
    {"role": "system", "content": sme1_response_text + sme2_response_text,
     "role": "user:", "content": base_premise
    }
  ]
)

summarizer_response_text = summarizer_response.choices[0].message.content
print(f"{Fore.YELLOW}The summarizer says: {summarizer_response_text}{Fore.RESET}\n")
'''