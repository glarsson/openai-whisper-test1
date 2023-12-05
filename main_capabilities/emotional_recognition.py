from queue import Queue
from threading import Thread
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from colorama import Fore, Style
from collections import deque

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion", legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-emotion")

target_word = ""
past_context = ""
future_context = ""
context_length = 5

# continuous buffer of context words
wordbuffer_index = 0
wordbuffer = []

# Initialize a dictionary to store the word ratings
word_ratings = {}

# Create a queue to hold the words
word_queue = Queue()

def get_emotion():
    global wordbuffer
    global target_word
    global past_context
    global future_context
    global wordbuffer_index

    while not word_queue.empty():
        # Wait for a word to be added to the queue
        word = word_queue.get()

        # Add the word to the buffer
        wordbuffer.append(word)

        # If the buffer has enough words for context_length, past_context, target_word and future_context
        if len(wordbuffer) >= (context_length * 2 + 1):
            # Save the first context_length words to past_context
            past_context = wordbuffer[:context_length]
            
            # Save the next word as the target_word
            target_word = wordbuffer[context_length]

            # Save the next context_length words to future_context
            future_context = wordbuffer[context_length + 1 : context_length * 2 + 1]

            # Call the classifier function
            classifier(past_context, target_word, future_context)

            # Clear the buffer
            wordbuffer = []

        # Mark the task as done
        word_queue.task_done()


def classifier(past, target, future):
    input_text = f"emotion: {' '.join(past)} {target} {' '.join(future)} </s>"
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    # Generate the output using the model
    outputs = model.generate(input_ids, max_new_tokens=20)
    # Decode the output and return the emotion
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if decoded_output == "joy":
        print(f"{Fore.YELLOW}{decoded_output}{Style.RESET_ALL}")
    elif decoded_output == "anger":
        print(f"{Fore.RED}{decoded_output}{Style.RESET_ALL}")
    elif decoded_output == "sadness":
        print(f"{Fore.BLUE}{decoded_output}{Style.RESET_ALL}")
    elif decoded_output == "fear":
        print(f"{Fore.MAGENTA}{decoded_output}{Style.RESET_ALL}")
    elif decoded_output == "surprise":
        print(f"{Fore.CYAN}{decoded_output}{Style.RESET_ALL}")
    elif decoded_output == "love":
        print(f"{Fore.LIGHTMAGENTA_EX}{decoded_output}{Style.RESET_ALL}")
    elif decoded_output == "neutral":
        print(f"{Fore.WHITE}{decoded_output}{Style.RESET_ALL}")
    else:
        print(f"")

# Create a thread that will run the get_emotion function
t = Thread(target=get_emotion)

# Start the thread
t.start()

# Open the file and read the text
with open('output/voice.txt', 'r', encoding='utf-8') as file:
    listen_output_text = deque(file, 1)[0]

# Now you can add words to the queue and they will be processed by the get_emotion function
for word in listen_output_text.split():
    word_queue.put(word)

# Wait for all the words to be processed
t.join()