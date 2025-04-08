import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


MODEL_NAME = "t5-large" 
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)


def read_input(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()
2
def generate_summary(text, num_lines=3, words_per_line=15, min_length=50, length_penalty=2.0):
    max_length = num_lines * words_per_line 
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=length_penalty, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def write_summary(file_path, summary):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(summary)

INPUT_FILE = "input.txt"
OUTPUT_FILE = "summary.txt"

num_lines = int(input("How many lines of summary text would you like? "))

text = read_input(INPUT_FILE)

summary = generate_summary(text, num_lines=num_lines)

write_summary(OUTPUT_FILE, summary)

print("Summary saved to", OUTPUT_FILE)
