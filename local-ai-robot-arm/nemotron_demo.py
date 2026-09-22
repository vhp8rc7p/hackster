import torch
import nemo.collections.asr as nemo_asr

print("Loading Nemotron-Speech-Streaming-0.6B...")

# Load the model directly from HuggingFace
# Note: For large models, this may take a few minutes to download the weights the first time.
model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/nemotron-speech-streaming-en-0.6b")

# If you want to transcribe an audio file, you can use the transcribe() method.
# For example, to transcribe the audio file we just found in the folder:
# audio_path = "Huafa North Road.m4a"
# transcripts = model.transcribe([audio_path])
# print("Transcription:", transcripts[0])

print("Model loaded successfully!")
print("Ready to be integrated into your project.")
