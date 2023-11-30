import queue
import time
import torch
import cv2
import os
import threading
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import threading

# large model
#processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
#model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

# base model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

def image_to_text():
    timeout = 5  # Timeout after 5 seconds
    poll_interval = 0.1  # Poll every 100ms
    counter = 0
    image_path = 'temp/vision.png'

    raw_image = Image.open(image_path).convert('RGB')

    # Unconditional image captioning
    print("inputs...")
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
    print("output...")
    out = model.generate(**inputs, max_new_tokens=20)
    print("image description should come here:")
    print(processor.decode(out[0], skip_special_tokens=True))
    print("done!")

image_to_text()


# Start the webcam capture and image processing in separate threads
#webcam_thread = threading.Thread(target=webcam_capture)
#processing_thread = threading.Thread(target=image_to_text)

#webcam_thread.start()
#processing_thread.start()
