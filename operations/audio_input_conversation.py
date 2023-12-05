import pyaudio
import wave
import time

chunk = 1024
format = pyaudio.paInt16
channels = 1
rate = 44100
record_seconds = 5
flush_interval = 5

file_index = 0

p = pyaudio.PyAudio()
stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

print("Recording...")
frames = []
start_time = time.time()

def record_audio_conversation():
    global file_index
    global frames
    global start_time
    global stream
    global chunk
    global format
    global channels
    global rate
    global record_seconds
    global flush_interval
    while True:

        data = stream.read(chunk)
        frames.append(data)

        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= flush_interval:
            output_filename = f"output/audio_input_convert_array_to_wave-output_{int(file_index)}.wav"
            wf = wave.open(output_filename, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
            wf.close()

            frames = []
            start_time = current_time
            print(f"Flushed buffer to {output_filename}")
            # Increment the file index
            file_index += 1
