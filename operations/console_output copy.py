from colorama import Fore
from colorama import Style
import colorama
import re
import logging
from datetime import datetime
import json
from collections import OrderedDict
import os
import nltk

colorama.init()
nltk.download('punkt')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

c_handler = logging.FileHandler('output/log.txt')
c_handler.setLevel(logging.INFO)
c_format = logging.Formatter('%(message)s')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)

style_file_path = 'console_config\\style_triggers.json'
color_file_path = 'console_config\\color_triggers.json'
output_color_file = 'console_config\\color_triggers.json'

def load_triggers_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            triggers = json.load(file)
        return triggers
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file at {file_path}")
        return {}

def concatenate_and_deduplicate(json_files, output_file):
    combined_triggers = OrderedDict()
    multi_word_triggers = set()  # New set to store multi-word triggers

    for file_path in json_files:
        try:
            with open(file_path, 'r') as file:
                triggers = json.load(file, object_pairs_hook=OrderedDict)
                # Strip whitespaces from keys
                triggers = {key.strip(): value for key, value in triggers.items()}
                combined_triggers.update(triggers)

                # Identify and store multi-word triggers
                for key in triggers.keys():
                    if ' ' in key:
                        multi_word_triggers.add(key.lower())

        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in file at {file_path}")

    with open(output_file, 'w') as output:
        json.dump(combined_triggers, output, indent=2)

    return multi_word_triggers

# We're splitting the files for obvious reasons here really (and deduplicating here as well should something slip through in the files)
# The idea is to get as many words that fit the sentiment as possible, this is not only a cool visual effect since we can apply STYLE too
# but it's basically a rudimentary sentiment analysis as well (that require pretty much no compute power what so ever).
#
# List of color trigger files for different sentiment levels
files_to_concatenate = [
    'console_config\\color_very_positive.json',
    'console_config\\color_positive.json',
    'console_config\\color_neutral.json',
    'console_config\\color_negative.json',
    'console_config\\color_very_negative.json',
    'console_config\\color_gangster_shit_haha.json'
]



concatenate_and_deduplicate(files_to_concatenate, output_color_file)

def apply_trigger(words, triggers, multi_word_triggers, attribute, default_value):
    result = []
    current_token = []
    max_attempts = 3  # Set the maximum number of attempts to find a match

    i = 0
    while i < len(words):
        current_token.append(words[i])
        current_token_str = ' '.join(current_token).lower()

        if current_token_str in triggers:
            value = triggers[current_token_str]
            if value is not None:
                color = getattr(attribute, value, None)
                if color is not None:
                    result.append(color + ' '.join(current_token) + Style.RESET_ALL)
                else:
                    result.append(default_value + ' '.join(current_token))
            current_token = []
        elif current_token_str in multi_word_triggers:
            for j in range(i + 1, min(i + max_attempts, len(words))):  # Use a sliding window
                current_token.append(words[j])
                current_token_str = ' '.join(current_token).lower()
                if current_token_str in triggers:  # Use triggers, not multi_word_triggers
                    value = triggers[current_token_str]
                    color = getattr(attribute, value, None)
                    if color is not None:
                        result.append(color + ' '.join(current_token) + Style.RESET_ALL)
                    else:
                        result.append(default_value + ' '.join(current_token))
                    current_token = []
                    i = j  # Move the main loop index to the last matched word
                    break  # Found a match, exit the loop
        else:
            result.append(default_value + words[i])

        i += 1

    return ' '.join(result)

def stylize_console_output(transcription_string, color_triggers, multi_word_triggers):
    words = transcription_string.lower().split()

    # Run color_triggers
    color_result = apply_trigger(words, color_triggers, multi_word_triggers, Fore, '')
    print(color_result)
    print(Style.RESET_ALL)
    # log the raw result, why not.
    current_time = datetime.now().strftime("%H:%M:%S.%f")
    logger.info(f"{current_time} [stylize_console_output] {color_result}")

    # Run style_triggers
    #style_result = apply_trigger(words, style_triggers, Style, 'RESET')  # 'RESET_ALL' should be 'RESET'
    # and PRINT that shit! That was the whole point :D
    #print(style_result)
    # and I guess we reset cause otherwise we'd be stuck at the last style and color applied forever
    #print(Style.RESET_ALL)
    


# Example song lyrics to analyse:
#with open('temp/eminem-lose-yourself.txt', 'r') as f:
#with open('temp/eminem-soldier.txt', 'r') as f:
with open('temp/lupe-fiasco-american-t.txt', 'r') as f:
    transcription = f.read()


#transcription = "you can go wild here too if you wish."
# Example usage
#color_result = apply_trigger(words, color_triggers, Fore, 'WHITE')
#style_result = apply_trigger(words, style_triggers, Style, 'RESET_ALL')
   
# Example usage
#multi_word_triggers = concatenate_and_deduplicate(files_to_concatenate, output_color_file)
#stylize_console_output(transcription, multi_word_triggers)

# Example usage
color_triggers = load_triggers_from_file(color_file_path)
multi_word_triggers = concatenate_and_deduplicate(files_to_concatenate, output_color_file)
stylize_console_output(transcription, color_triggers, multi_word_triggers)

#stylize_console_output(transcription)