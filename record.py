import sounddevice as sd
import numpy as np
import soundfile as sf
import time

# Configuration parameters
threshold = 0.045  # RMS threshold for silence detection (adjust as needed)
silence_duration = 2  # Seconds of silence to trigger stop
sample_rate = 44100  # Sampling rate
channels = 1         # Number of audio channels

# Global variables
audio_buffer = []
last_loud_time = time.time()
is_recording = True

def audio_callback(indata, frames, time_info, status):
    global last_loud_time, is_recording, audio_buffer
    
    # Calculate RMS of current audio block
    rms = np.sqrt(np.mean(np.square(indata)))
    
    # Append audio data to buffer (regardless of silence)
    audio_buffer.append(indata.copy())
    
    if rms > threshold:
        # Reset silence timer when sound is detected
        last_loud_time = time.time()
    else:
        # Stop recording if silence persists beyond duration
        if time.time() - last_loud_time > silence_duration:
            is_recording = False
            raise sd.CallbackStop

def record_audio(filename):
    # Create input stream
    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        callback=audio_callback
    )

    print("Recording started...")
    with stream:
        while is_recording:
            sd.sleep(100)  # Keep the stream active until recording stops

    # Save recorded audio to file
    if len(audio_buffer) > 0:
        audio_data = np.concatenate(audio_buffer, axis=0)
        sf.write(filename, audio_data, sample_rate)
        print(f"Recording saved to {filename}")
        print(f"Duration: {len(audio_data)/sample_rate:.2f} seconds")
    else:
        print("No audio recorded")