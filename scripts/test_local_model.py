from gpt4all import GPT4All
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "orca_mini_3b.Q3_K_M.gguf")

print("Loading local model from:", MODEL_PATH)

model = GPT4All(MODEL_PATH, allow_download=False)

with model.chat_session():
    prompt = "Create 1 medium-difficulty MCQ with 4 options about Python functions."
    response = model.generate(prompt, max_tokens=200)

print("\nGenerated MCQ:")
print(response)











