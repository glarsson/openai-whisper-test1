import os
import time
import unicodedata
from faster_whisper import WhisperModel
from colorama import Fore

# whisper-faster inits
# model_size can be "small" or "large" - obvious tradeoffs
model_size = "small"
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

print("voice.py: model loaded")

# beam_size:
# Larger = more accurate, but slower performance (it considers more possibilities and thus LLM-style finding the best path)
# Smaller = less accurate, but faster performance
# 5 is a good balance between speed and accuracy for the small model, but 3 seems to do the trick for now

whisper_beam_size = 10

def remove_non_unicode(text):
    # Function to remove non-unicode characters from the text
    return ''.join(char for char in text if unicodedata.category(char)[0] != 'C')

def transcribe_audio():
    transcriber_file_index = 0
    while True:
        #print("index %s" % (transcriber_file_index))

        # Create a temporary "pointer" using the current index
        temp_file_name = f"output/output_{transcriber_file_index}.wav"
        
        # check that temp_file_name exists on the file system, otherwise wait for it to be written
        while not os.path.exists(temp_file_name) or os.path.getsize(temp_file_name) < 10240:
            time.sleep(0.1)

        # Transcribe the audio
        segments, info = model.transcribe(temp_file_name, beam_size=whisper_beam_size)

        # Initialize an empty list to hold the transcriptions
        transcriptions = []

        # Add the segment transcriptions to the list

        for segment in segments:
            start_time = segment.start
            end_time = segment.end
            text = segment.text
            transcriptions.append((start_time, end_time, text))
        #print(f"Transcription completed for internal index {transcriber_file_index}.")            

        transcription_string = ' '.join(text for _, _, text in transcriptions)
        
        # Remove non-unicode characters
        transcription_string = remove_non_unicode(transcription_string)

        # increment local index        
        transcriber_file_index += 1

        # Print the transcription string
        unwanted_words = ["you", "Thank you.", "Thanks for watching!"]
        if not any(word in transcription_string and transcription_string.count(word) == 1 for word in unwanted_words):
            print(Fore.YELLOW + f"[voice] {transcription_string}" + Fore.RESET)
            # append output to file
            with open('output/voice.txt', 'a', encoding='utf-8') as f:
                f.write("[voice] " + transcription_string + "\n")
