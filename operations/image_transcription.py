from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import time
from colorama import Fore
import torch
from PIL import Image
import logging

# Create a custom logger
logger_image_to_text = logging.getLogger(__name__)
# Set the logger's level to INFO
logger_image_to_text.setLevel(logging.INFO)
# Create handlers
c_handler = logging.FileHandler('log.txt')
c_handler.setLevel(logging.INFO)
# Create formatters and add it to handlers
c_format = logging.Formatter('%(message)s')
c_handler.setFormatter(c_format)
# Add handlers to the logger
logger_image_to_text.addHandler(c_handler)

# large model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

# base model
#processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

def image_to_text():
    poll_interval = 0.1  # Poll every 100ms
    counter = 0
    image_path = 'temp/vision2.png'
    logger_image_to_text.info("whisper-faster model loaded")

    while True:
        while True:
            if os.path.exists(image_path) and os.stat(image_path).st_size > 0:
                time.sleep(0.2)
                raw_image = Image.open(image_path).convert('RGB')
                break
            else:
                time.sleep(poll_interval)
                counter += poll_interval

        # Start the timer
        start_time = time.time()

        # Unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
        out = model.generate(**inputs, max_new_tokens=30)
        current_output = "[vision]" + processor.decode(out[0], skip_special_tokens=True) + "\n"

        # Stop the timer
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        # Log the elapsed time
        logger_image_to_text.info(f"Transformers processing took {elapsed_time} seconds ({elapsed_time * 1000} milliseconds)")


        # Check if the file exists and is not empty
        if os.path.exists('vision.txt') and os.path.getsize('vision.txt') > 0:
            with open('vision.txt', 'r') as f:
                last_line = f.readlines()[-1].strip()  # Read the last line of the file

            # Compare the current output with the last line of the file
            if current_output != last_line:
                # If they are not the same, write the current output to the file
                with open('vision.txt', 'a') as f:
                    f.write(current_output + '\n')
        else:
            # If the file does not exist or is empty, write the current output to the file
            with open('vision.txt', 'a') as f:
                f.write(current_output + '\n')


        #print("image description should come here:")
        print(Fore.GREEN + current_output + Fore.RESET)


        # appent output to file
        with open('output/vision.txt', 'a', encoding='utf-8') as f:
            f.write("[vision] " + current_output + "\n")
            f.close()

        # Delete the file after processing
        os.remove(image_path)