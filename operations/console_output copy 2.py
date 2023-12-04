import json
import re
from colorama import Fore, Style

def extract_multi_word_keys(json_path):
    with open(json_path, 'r') as file:
        color_triggers = json.load(file)

    # Convert keys to lowercase and filter non-ASCII characters
    multi_word_keys = [
        re.sub(r'[^a-zA-Z0-9 ]', '', key.lower())
        for key in color_triggers.keys()
        if ' ' in key
    ]
    return multi_word_keys

# Example usage:
#multi_word_keys = extract_multi_word_keys('console_config/color_triggers.json')

#print(multi_word_keys)

from colorama import Fore, Style

# Map your custom color names to colorama color names
COLOR_MAP = {
    "black": Fore.BLACK,
    "red": Fore.RED,
    "green": Fore.GREEN,
    "yellow": Fore.YELLOW,
    "blue": Fore.BLUE,
    "magenta": Fore.MAGENTA,
    "cyan": Fore.CYAN,
    "white": Fore.WHITE,
    "lightblack": Fore.LIGHTBLACK_EX,
    "lightred": Fore.LIGHTRED_EX,
    "lightgreen": Fore.LIGHTGREEN_EX,
    "lightyellow": Fore.LIGHTYELLOW_EX,
    "lightblue": Fore.LIGHTBLUE_EX,
    "lightmagenta": Fore.LIGHTMAGENTA_EX,
    "lightcyan": Fore.LIGHTCYAN_EX,
    "lightwhite": Fore.LIGHTWHITE_EX
}

def create_multi_word_index(json_path):
    with open(json_path, 'r') as file:
        color_triggers = json.load(file)

    # Convert keys to lowercase and filter non-ASCII characters
    multi_word_sentences_index = {}

    for key in color_triggers.keys():
        if ' ' in key:
            cleaned_key = re.sub(r'[^a-zA-Z0-9 ]', '', key.lower())
            words = cleaned_key.split()
            for i in range(len(words) - 1):
                prefix = ' '.join(words[:i + 1])
                if prefix not in multi_word_sentences_index:
                    multi_word_sentences_index[prefix] = set()

                multi_word_sentences_index[prefix].add(words[i + 1])

    return multi_word_sentences_index

# Example usage:
multi_word_sentences_index = create_multi_word_index('console_config/color_triggers.json')
print(multi_word_sentences_index)

def apply_color(word, color):
    # Use the mapped color or default to white if not found
    colorama_color = COLOR_MAP.get(color.lower(), Fore.WHITE)
    colored_word = colorama_color + word + Style.RESET_ALL
    return colored_word

# Example usage:
#colored_word = apply_color("example", "green")
#print(colored_word)

def process_words(input_stream, color_triggers, multi_word_indexes):
    iterator = iter(input_stream)
    
    for word in iterator:
        if word in multi_word_indexes:
            # Check if the next words form a multi-word key
            next_word = next(iterator, '')
            multi_word_key = ' '.join([word, next_word])
            if multi_word_key in color_triggers:
                color = color_triggers[multi_word_key]
                print(apply_color(multi_word_key, color))
                continue

        # Process individual words
        if word in color_triggers:
            color = color_triggers[word]
            print(apply_color(word, color))

# Example usage:
color_triggers = {"apple": "red", "banana": "yellow", "dark chocolate": "brown"}
multi_word_indexes = set(['dark'])
input_stream = ["dark", "chocolate", "banana", "apple", "dark", "chocolate"]

process_words(input_stream, color_triggers, multi_word_indexes)
