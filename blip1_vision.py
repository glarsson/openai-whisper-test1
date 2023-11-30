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
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

# base model
#processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

# Create a lock for the webcam writing process
lock = threading.Lock()

# Create a Queue object to use as the buffer
frame_buffer = queue.Queue(maxsize=1)  # Adjust the maxsize as needed

def image_to_text():
    poll_interval = 0.1  # Poll every 100ms
    counter = 0
    image_path = 'temp/vision2.png'
    while True:
        while True:
            if os.path.exists(image_path) and os.stat(image_path).st_size > 0:
                time.sleep(0.5)
                raw_image = Image.open(image_path).convert('RGB')
                break
            else:
                time.sleep(poll_interval)
                counter += poll_interval

        # Unconditional image captioning
        #print("inputs...")
        inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
        #print("output...")
        out = model.generate(**inputs, max_new_tokens=50)
        #print("image description should come here:")
        print(Fore.GREEN + processor.decode(out[0], skip_special_tokens=True) + Fore.RESET)
        #print("done!")

        # Delete the file after processing
        os.remove(image_path)

def webcam_capture():
    cap = cv2.VideoCapture(cam_device_index)
    
    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Unable to open camera")
        return
    
    while True:
        ret, frame = cap.read()
        
        # Check if the frame was read successfully
        if not ret:
            break

        # Convert the OpenCV BGR image to RGB (PIL format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the image to 800x600
        resized_frame = cv2.resize(rgb_frame, (1280, 720))

        # Acquire the lock before writing to the file
        with lock:
            # Write the image to the output file
            output_filename = 'temp/vision.png'
            ret, buffer = cv2.imencode(".png", cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))
            if ret:
                with open(output_filename, 'wb') as out_file:  # Define the out_file variable
                    out_file.write(buffer)
                    out_file.close()  # Close the file after writing
                    if not os.path.exists('temp/vision2.png'):
                        shutil.copy(output_filename, 'temp/vision2.png')
                    time.sleep(0.2)

            # Only copy the file if it does not exist - this is to prevent the processing thread from reading an empty file
            # this whole thing is a really poor mans solution to threading and please don't judge me, it's only a PoC.


    # Release the camera when the loop is exited
    cap.release()

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

