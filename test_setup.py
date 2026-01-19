import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import streamlit as st
from gpt4all import GPT4All  # Replacing OpenAI

# Test: print that all libraries loaded
print("All libraries imported successfully!")

# Optional: Quick GPT4All test
try:
    # Load your GPT4All model (replace with your actual .bin filename)
    model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")
    prompt = "Write 1 multiple-choice question (4 options) about Python programming."
    response = model.generate(prompt)
    print("GPT4All test successful! Generated MCQ:")
    print(response)
except Exception as e:
    print("GPT4All test failed. Error:", e)
