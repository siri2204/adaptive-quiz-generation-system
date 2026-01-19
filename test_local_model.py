from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Path to your downloaded model
model_path = "./orca_mini_3b.Q3_K_M.gguf"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Create a text-generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Test prompt
prompt = "Write 1 multiple-choice question (4 options) about Python programming."

# Generate output
outputs = generator(prompt, max_length=200, do_sample=True)
print("\nGenerated MCQ:")
print(outputs[0]['generated_text'])
