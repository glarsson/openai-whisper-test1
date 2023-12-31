import threading
from colorama import Fore

import sys
sys.path.append('C:\SOURCE\GERRY\openai-whisper-test1')

from operations.image_capture import webcam_capture
from operations.image_transcription import image_to_text


# Start the webcam capture and image processing in separate threads
webcam_thread = threading.Thread(target=webcam_capture)
processing_thread = threading.Thread(target=image_to_text)

webcam_thread.start()
processing_thread.start()