import queue
import shutil
import time
import torch
import cv2
import os
import threading
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import threading
import pywintypes
import win32com.client
from colorama import Fore

import image_operations
import image_transcription

'''
def get_device_names():
    wmi = win32com.client.GetObject ("winmgmts:")
    devices = wmi.InstancesOf ("Win32_PnPEntity")
    device_names = []
    for device in devices:
        if device.Name is not None and "cam" in device.Name.lower():
            device_names.append(device.Name)
    return device_names

device_names = get_device_names()

for i, device_name in enumerate(device_names):
    print(f"Device {i}: {device_name}")
'''

cam_device_index = 3

# large model
#processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
#model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

# base model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

# Create a lock for the webcam writing process
lock = threading.Lock()

# Create a Queue object to use as the buffer
frame_buffer = queue.Queue(maxsize=1)  # Adjust the maxsize as needed

# Start the webcam capture and image processing in separate threads
webcam_thread = threading.Thread(target=webcam_capture)
processing_thread = threading.Thread(target=image_to_text)

webcam_thread.start()
processing_thread.start()

# hit the brakes (good for making sure the cam doesn't get locked/screwed up)
try:
    webcam_thread.join()
    #cap.release()
    webcam_thread.join()
    processing_thread.join()
except KeyboardInterrupt:
    pass

