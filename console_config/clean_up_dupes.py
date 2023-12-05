from colorama import Fore, Style
import json
import glob
import os
import re

def combine_color_files():
    directory = 'console_config'
    output_file = f'{directory}/color_triggers.json'

    # Get a list of all color_*.json files in the directory
    file_paths = glob.glob(f'{directory}/*.json')

    combined_data = {}

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
            for key, value in data.items():
                # Convert key to lowercase and remove non-ASCII characters
                key = ''.join(c for c in key.lower() if c.isascii())
                # Convert value to lowercase and remove non-ASCII characters
                value = ''.join(c for c in value.lower() if c.isascii())
                combined_data[key] = value

    # Write the combined data to the output file
    with open(output_file, 'w') as file:
        json.dump(combined_data, file, indent=4)

combine_color_files()

def get_multi_word_triggers(color_triggers):
    multi_word_triggers = {}
    for key, value in color_triggers.items():
        words = key.split()
        if len(words) > 1:
            multi_word_triggers[key] = value
    return multi_word_triggers

def apply_color(word_from_array, color):
    COLOR_MAP = {
        "black": Fore.BLACK,
        "red": Fore.RED, # picked for Very Negative
        "green": Fore.GREEN, # picked for Very Positive
        "yellow": Fore.YELLOW, # picked for Neutral
        "blue": Fore.BLUE,
        "magenta": Fore.MAGENTA,
        "cyan": Fore.CYAN,
        "white": Fore.WHITE, # default color for unset words
        "lightblack": Fore.LIGHTBLACK_EX,
        "lightred": Fore.LIGHTRED_EX, # picked for Negative
        "lightgreen": Fore.LIGHTGREEN_EX, # picked for Positive
        "lightyellow": Fore.LIGHTYELLOW_EX,
        "lightblue": Fore.LIGHTBLUE_EX,
        "lightmagenta": Fore.LIGHTMAGENTA_EX,
        "lightcyan": Fore.LIGHTCYAN_EX,
        "lightwhite": Fore.LIGHTWHITE_EX
    }

    colorama_color = COLOR_MAP.get(color.lower(), Fore.WHITE)
    colored_word = colorama_color + word_from_array + Style.RESET_ALL
    return colored_word

def process_words(words, color_triggers):
    i = 0
    while i < len(words):
        word = words[i]
        multi_word_match = False

        for key, value in color_triggers.items():
            key_words = key.split()

            if key_words and word == key_words[0] and words[i:i+len(key_words)] == key_words:
                print(apply_color(' '.join(key_words), value), end=" ")
                i += len(key_words)
                multi_word_match = True
                break

        if not multi_word_match:
            colored_word = print_word_with_color(word, color_triggers)
            print(colored_word, end=" ")
            i += 1

def print_word_with_color(word, color_triggers):
    word = word.replace('\n', ' ').replace('\r', '')
    word = word.replace('\n', ' ')
    if word in color_triggers:
        colored_word = apply_color(word, color_triggers[word])
    else:
        colored_word = word
    return colored_word

def read_and_process_input_file(file_path, color_triggers):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Remove newline characters
    content = content.replace('\n', ' ').replace('\r', ' ')

    # Split the string into lines
    lines = content.split('\n')

    # Filter out the empty lines
    lines = [line for line in lines if line.strip() != '']

    # Join the lines back together
    content = '\n'.join(lines)

    # Convert to lowercase
    content = content.lower()

    # Keep only ASCII letters, periods, commas, and whitespace
    content = re.sub(r'[^a-z.,\s]', '', content)

    words = content.split()

    process_words(words, color_triggers)

# Load color triggers
with open('console_config/color_triggers.json', 'r') as file:
    color_triggers = json.load(file)

# Process input file
read_and_process_input_file('temp/text-to-transcribe-to-color.txt', color_triggers)
