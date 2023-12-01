from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import time
from colorama import Fore
import torch
from PIL import Image

# large model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

# base model
#processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

print("vision.py: model loaded")

def image_to_text():
    poll_interval = 0.1  # Poll every 100ms
    counter = 0
    image_path = 'temp/vision2.png'
    while True:
        while True:
            if os.path.exists(image_path) and os.stat(image_path).st_size > 0:
                time.sleep(0.2)
                raw_image = Image.open(image_path).convert('RGB')
                break
            else:
                time.sleep(poll_interval)
                counter += poll_interval

        # Unconditional image captioning
        #print("inputs...")
        inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
        #print("output...")
        out = model.generate(**inputs, max_new_tokens=30)
        #print("image description should come here:")
        print(Fore.GREEN + "[vision] " + processor.decode(out[0], skip_special_tokens=True) + Fore.RESET)
        #print("done!")

        # appent output to file
        with open('output/vision.txt', 'a', encoding='utf-8') as f:
            f.write("[vision] " + processor.decode(out[0], skip_special_tokens=True) + "\n")
            f.close()

        # Delete the file after processing
        os.remove(image_path)