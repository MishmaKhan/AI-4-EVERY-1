# Use a pipeline as a high-level helper
from transformers import pipeline
import torch

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-medium")

model_path = ("../../Models/models--openai--whisper-medium/"
              "snapshots/abdf7c39ab9d0397620ccaea8974cc764cd0953e")
 
device = "cuda" if torch.cuda.is_available() else "cpu"
 
 
pipe = pipeline("automatic-speech-recognition",
                model=model_path, chunk_length_s=30, device=device)
 
out = pipe("../../Resources/test.mp3")["text"]
print(out)