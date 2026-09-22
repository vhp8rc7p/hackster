from mlx_audio.stt import load

print("Loading Nemotron-Speech-Streaming via MLX...")

# Load the MLX port of the Nemotron model
model = load("mlx-community/nemotron-3.5-asr-streaming-0.6b")

print("Model loaded successfully!")

# To transcribe the audio file we found earlier:
# audio_path = "Huafa North Road.m4a"
# print(f"Transcribing {audio_path}...")
# result = model.generate(audio_path)
# print("Transcription:", result.text)

print("Ready to be integrated into your project.")
