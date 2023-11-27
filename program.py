from openai import OpenAI
from colorama import Fore
#import sounddevice as sd
import numpy as np
#from scipy.io.wavfile import write
import keyboard

# Read API key from a file
with open('secret_apikey.txt', 'r') as file:
    api_key = file.read().strip()

client = OpenAI(api_key=api_key)

# Our first AI 'assistant' role and its speciality
sme1_specialization = "psychologist"
openai_sme1 = f"As an expert {sme1_specialization}, this is two sentences on that topic:"

# Our second AI 'assistant' role and its speciality
sme2_specialization = "shaman"
openai_sme2 = f"As an expert {sme2_specialization}, this is two sentences on that topic:"

# The base premise of what we are trying to do
base_premise = f"""
The following will be a user started, but AI led conversation - multi input/output from different assistants to reach a final conclusion.
It begins with the user asking a question which we will pipe to different openai assistants.

Our two experts are {sme1_specialization} and {sme2_specialization}
"""
print(base_premise)

# Get user input
user_input = input("Please enter your question: ")


# we're gonna start to do some whisper stuff later, it seems troublesome to get audio recorded in python and windows for some reason.
# print(f"{Fore.WHITE}Whisper things you said: {user_input}{Fore.RESET}\n")

# first sme response
sme1_response = client.chat.completions.create(
  model="gpt-3.5-turbo-16k-0613",
  max_tokens=512,
  messages=[
    {"role": "assistant", "content": openai_sme1 + user_input }
    ]
)

sme1_response_text = sme1_response.choices[0].message.content
print(f"{Fore.CYAN}The {sme1_specialization} says: {sme1_response_text}{Fore.RESET}\n")

# second sme response
sme2_response = client.chat.completions.create(
  model="gpt-3.5-turbo-16k-0613",
  max_tokens=512,
  messages=[
    {"role": "assistant", "content": openai_sme2 + user_input }
    ]
)

sme2_response_text = sme2_response.choices[0].message.content
print(f"{Fore.MAGENTA}The {sme2_specialization} says: {sme2_response_text}{Fore.RESET}\n")

# initialize summarizer
init_summarizer = "please summarize the following two statements as condensed as possible:"
openai_summarizer = init_summarizer +  openai_sme1 + ": " + sme1_response_text + ". " + openai_sme2 + ": " + sme2_response_text

# summarizer response
summarizer_response = client.chat.completions.create(
  model="gpt-3.5-turbo-16k-0613",
  max_tokens=1024,
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
  max_tokens=4096,
  messages=[
    {"role": "system", "content": "Please discuss amongst yourselves, the answers provided:" + summarizer_response_text},
    {"role": "assistant", "content": sme1_specialization + sme1_response_text},
    {"role": "assistant", "content": sme2_specialization + sme2_response_text}
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