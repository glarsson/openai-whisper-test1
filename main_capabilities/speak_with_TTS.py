"""
(X) NOTES
(X1) to get list of models
 - run "tts --list_models" in command line
 - or open "venv\lib\site-packages\tts\.models.json"
(X2) if model has multiple speakers, check tts\model\speaker_ids.json
(X3) edit "tts\model\config.json" to control voices
 - length_scale = speed (more is slower)
 - noise_scale  = speech variation (more is dynamic)
"""

# (A) LOAD MODULES
import wave
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import os
import pyaudio

def text_to_audio(SET_TXT):
    # (B) SETTINGS
    PATH_BASE = os.path.dirname(__file__)

    # We'll set this in the loop
    #SET_TXT = os.path.join(PATH_BASE, "narrate.txt")

    SET_SAVE = os.path.join(PATH_BASE, "output/tts_output.wav")
    SET_MODEL = "tts_models/en/vctk/vits"
    SET_SPEAKER = "p274"

    # (C) MODEL MANAGER
    manager = ModelManager(
      models_file = PATH_BASE + "\\env\\Lib\\site-packages\\TTS\\.models.json",
      output_prefix = PATH_BASE,
      progress_bar = True
    )

    #debug
    print(PATH_BASE)

    model_path, config_path, model_item = manager.download_model(SET_MODEL)
    if model_item["default_vocoder"] is None:
      voc_path = None
      voc_config_path = None
    else:
      voc_path, voc_config_path, _ = manager.download_model(model_item["default_vocoder"])

    # (D) SYNTHESIZER
    syn = Synthesizer(  
      tts_checkpoint = model_path,
      tts_config_path = config_path,
      vocoder_checkpoint = voc_path,
      vocoder_config = voc_config_path,
      use_cuda = True
    )

    while True:

        # (E) OUTPUT
        output = syn.tts(
          text = open(SET_TXT, "r").read(),
          speaker_name = SET_SPEAKER
        )
        syn.save_wav(output, SET_SAVE)



        # playback
        # Open the output file
        wf = wave.open(SET_SAVE, 'rb')

        # Create a PyAudio object
        p = pyaudio.PyAudio()

        # Open a stream
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        # Read data from the file and play it
        data = wf.readframes(1024)
        while len(data) > 0:
            stream.write(data)
            data = wf.readframes(1024)

        # Close the stream and the PyAudio object
        stream.stop_stream()
        stream.close()
        p.terminate()

        # also delete the output file that was processed
        os.remove(SET_SAVE)

    