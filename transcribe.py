import os, whisper
from dotenv import load_dotenv

load_dotenv()

def transcribe(audio_path):
    model = whisper.load_model(os.getenv("DEFAULT_WHISPER_MODEL"))

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio_trim = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio_trim, n_mels=model.dims.n_mels).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    result = model.transcribe(audio_path)

    # print the recognized text
    print(result['text'])
    return result['text']