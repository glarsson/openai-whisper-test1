
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
sme1_specialization = "English language professor"
# The base premise of what we are trying to do
base_premise_vision = f"""{sme1_specialization}. You are going to provide me with as much data as you possibly can in the format in which I desire."""

# Our second AI 'assistant' role and its speciality
sme2_specialization = "head of the English department at a prestigious university"
base_premise_rector = f"""{sme2_specialization}. You are going to analyze the content (words and classification) from the English language professor 
and provide feedback for him to research further if not all goals are met - we are looking to get at least 500 words in each category."""

goal = "We are looking to get at least 500 english non-duplicated words categorized in terms of how generally hostile or friendly they are, i.e. from Very Positive, Positive, Neutral, Negative and Very Negative"

# print(base_premise)

####
####

# First we're gonna have to actually watch the video and grab the content... :D

# So I will do that not in here but separately, later connect them.

####
####
'''
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
'''


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
     "role": "assistant", "content": "You will take both inputs from the two previous AI's and summarize them into a condensed final summary.",
     "role": "user", "content": sme1_response_text + sme2_response_text
    }
  ]
)

summarizer_response_text = summarizer_response.choices[0].message.content
print(f"{Fore.YELLOW}The summarizer says: {summarizer_response_text}{Fore.RESET}\n")
'''