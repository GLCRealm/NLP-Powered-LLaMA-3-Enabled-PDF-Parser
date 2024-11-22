from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import os
import subprocess
from transformers import AutoTokenizer
import requests
import re
import tkinter as tk
from tkinter import filedialog
from difflib import SequenceMatcher

# Step 1: Check if Poppler is installed
def is_poppler_installed():
    try:
        subprocess.run(["pdftoppm", "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return True  # Poppler exists but command returned an error


# Step 2: Extract text from a text-based PDF
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = "".join([page.extract_text() or "" for page in reader.pages])
        return text
    except Exception as e:
        print(f"Error reading PDF with PyPDF2: {e}")
        return ""


# Step 3: Extract text from image-based PDF
def extract_text_from_images(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        text = "".join([pytesseract.image_to_string(image) for image in images])
        return text
    except Exception as e:
        print(f"Error processing PDF as images: {e}")
        return ""


# Step 4: Extract text (handles both text-based and image-based PDFs)
def extract_text(pdf_path):
    if not is_poppler_installed():
        print("Poppler is not installed or not in PATH. Please install it.")
        return ""

    try:
        text = extract_text_from_pdf(pdf_path)
        if text.strip():
            return text
        print("No text found in PDF, attempting OCR...")
        return extract_text_from_images(pdf_path)
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""


# Step 5: Load tokenizer for the model
def load_tokenizer(model_name="meta-llama/Llama-3.2-3B-Instruct"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None


# Step 6: Tokenize text into manageable chunks
def tokenize_text(text, tokenizer, max_length=2048):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    return [tokenizer.decode(tokens[i:i + max_length], skip_special_tokens=True)
            for i in range(0, len(tokens), max_length)]


# Step 7: Preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.upper()  # Convert to uppercase


# Step 8: Query the LLaMA model
def query_llama(api_key, input_text, model_name="meta-llama/Llama-3.2-3B-Instruct"):
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": input_text, "parameters": {"temperature": 0.5, "max_new_tokens": 200}} # Adjust parameters as needed

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying LLaMA model: {e}")
        return None


# Step 9: Find relevant chunk for a question
def find_relevant_chunk(chunks, question, tokenizer, max_tokens=3000):

    chunk_scores = [
        (chunk, len(tokenizer(chunk)["input_ids"]), SequenceMatcher(None, chunk.lower(), question.lower()).ratio())
        for chunk in chunks
    ]
    chunk_scores.sort(key=lambda x: x[2], reverse=True)

    selected_chunks = []
    token_count = 0
    for chunk, tokens, _ in chunk_scores:
        if token_count + tokens > max_tokens:
            break
        selected_chunks.append(chunk)
        token_count += tokens

    return "\n".join(selected_chunks)


# Step 10: Answer a question using the model
def answer_question(api_key, chunks, question, tokenizer, max_context_tokens=3000):
    context = find_relevant_chunk(chunks, question, tokenizer, max_context_tokens)
    input_text = f"Context: {context}\nQuestion: {question}"

    response = query_llama(api_key, input_text)
    if response and "generated_text" in response[0]:
        return response[0]["generated_text"]
    return "Unable to generate an answer."


# Step 11: Browse for a PDF file
def browse():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(title="Select a PDF file", filetypes=[("PDF files", "*.pdf")])


# Main program
# Note: This code below is for demonstration purposes and may need to be adapted to your specific use case.


# if __name__ == "__main__":
#     pdf_path = browse()
#     if not pdf_path:
#         print("No file selected.")
#         exit()
#
#     api_key = "" # Add your API key here
#
#     if not os.path.exists(pdf_path):
#         print("PDF file not found.")
#         exit()
#
#     text = extract_text(pdf_path)
#     preprocessed_text = preprocess_text(text)
#
#     if not preprocessed_text.strip():
#         print("No valid text extracted.")
#         exit()
#
#     tokenizer = load_tokenizer()
#     if not tokenizer:
#         print("Failed to load tokenizer.")
#         exit()
#
#     chunks = tokenize_text(preprocessed_text, tokenizer)
#
#     while True:
#         question = input("\nAsk a question (or type 'exit' to quit): ")
#         if question.lower() == "exit":
#             break
#
#         answer = answer_question(api_key, chunks, question, tokenizer)
#         answer = answer.split(sep=question)
#         print(answer[1])