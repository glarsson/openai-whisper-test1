
from openai import OpenAI
from colorama import Fore
import numpy as np
import re

# Read API key from a file
with open('secret_apikey.txt', 'r') as file:
    api_key = file.read().strip()

client = OpenAI(api_key=api_key)

already_gathered_words = """

"""

# specify the OpenAI model to use:
# https://platform.openai.com/docs/models/gpt-3-5
# 16k token limit on "gpt-3.5-turbo-16k-0613"
# 16k token limit on "gpt-3.5-turbo-1106" (newest release as of dec-1-2023)
gpt_model = "gpt-3.5-turbo-1106"

RESEARCH_TOPIC = "positive words! POSITIVE THINKING IN ONE WORD"
primary_goal = f"PRIMARY GOAL: We are looking to get at least 500 of english unique words that fits the description: {RESEARCH_TOPIC}. Very important: words strictly related to {RESEARCH_TOPIC}."
secondary_goal = f"SECONDARY GOAL: Keep to a maximum of three words, preferrably one word per line, but up to three words per line is acceptable. IMPORTANT: create two versions of the words every time - one plural, one singular, of course on separate lines, and only input the word - nothing else, no line numbers, no indexes, no capital letters, ONLY the words in ASCII"


fuck_line_numbers = "do NOT add (line) numbers in front of the word, only input the single word - nothing else."



###############################################
################# SME1 ########################
###############################################
sme1_specialization = "superintelligent artificial intelligence"
sme1_base_premise = f"Produce as many {RESEARCH_TOPIC} words you can think of and put them in a single word per line - no line numbers or anything else, just a single word that resonates with {RESEARCH_TOPIC} - and NO DUPLICATES."
user_request = ""f"{RESEARCH_TOPIC} - those are the kind of words we want. Please format your output as a single word per line, or up to three words in a short sentence per line. Remember - only {RESEARCH_TOPIC}, as many words that are {RESEARCH_TOPIC} as you can come up with"""

stream = client.chat.completions.create(
    model=gpt_model,
    messages=[
        {"role": "system", "content": primary_goal + secondary_goal + sme1_specialization + sme1_base_premise,
         "role": "assistant", "content": (f"Your instructions are {sme1_base_premise}, {fuck_line_numbers}"),         
         "role": "user", "content": "give me 100 items."
    }], stream=True)

sme1_output = []

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print("\033[92m" + chunk.choices[0].delta.content + "\033[0m", end="")
        # Add the string to the list
        sme1_output.append(chunk.choices[0].delta.content)
        # Join the strings together to get the final result
        sme1_result = "".join(sme1_output)
with open('DATACOLLECTION_SME1_GPT3.5.txt', 'w') as file:
    file.write(sme1_result)

print()

###############################################
################# SME2 ########################
###############################################
sme2_specialization = "A Carl Sagan or Oppenheimer type super-ingelligent human being"
sme2_base_premise = f"We are looking to get 200 english unique words that fits the description: {RESEARCH_TOPIC}. It is ((VERY IMPORTANT)) that the words are strictly related to {RESEARCH_TOPIC} and ONLY ASCII WORDS, no emojis or other non-ASCII characters and no line numbers. IMPORTANT: create two versions of the words every time - one plural, one singular, of course on separate lines"

stream = client.chat.completions.create(
  model=gpt_model,
  messages=[
        {"role": "system", "content": primary_goal + secondary_goal + sme2_specialization + sme2_base_premise,
         "role": "assistant", "content": (f"Your instructions are {sme2_base_premise}, {fuck_line_numbers}"),
         "role": "user", "content": "give me 100 items."
    }], stream=True)

sme2_output = []

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print("\033[93m" + chunk.choices[0].delta.content + "\033[0m", end="")
        # Add the string to the list
        sme2_output.append(chunk.choices[0].delta.content)
        # Join the strings together to get the final result
        sme2_result = "".join(sme2_output)
with open('DATACOLLECTION_SME2_GPT3.5.txt', 'w') as file:
    file.write(sme2_result)

























#--------------------------
    
def concatenate_files(file1, file2, output_file):
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    # Remove numbering, convert to lowercase, and sanitize
    lines1 = [re.sub(r'[^a-z\s]', '', re.sub(r'^\d+\.\s*', '', line.lower().strip())) for line in lines1]
    lines2 = [re.sub(r'[^a-z\s]', '', re.sub(r'^\d+\.\s*', '', line.lower().strip())) for line in lines2]

    # Read existing lines in the output file
    with open(output_file, 'r', encoding='utf-8') as output:
        existing_lines = output.readlines()

    # Remove newline characters from existing lines
    existing_lines = [line.strip() for line in existing_lines]

    # Append unique lines to the output file
    with open(output_file, 'a', encoding='utf-8') as output:
        for line in lines1 + lines2:
            if line not in existing_lines:
                output.write(line + '\n')

# Example usage:
concatenate_files('DATACOLLECTION_SME1_GPT3.5.txt', 'DATACOLLECTION_SME2_GPT3.5.txt', f'GPT3.5-DATA_COLLECTION_{RESEARCH_TOPIC}.txt')