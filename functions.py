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

# Step 1: Check if poppler is installed
def is_poppler_installed():
    try:
        subprocess.run(["pdftoppm", "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return True  # Command ran but returned an error, meaning Poppler exists


# Step 2: Function to extract text from a text-based PDF
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF with PyPDF2: {e}")
        return ""


# Step 3: Function to extract text from image-based PDF
def extract_text_from_images(pdf_path):
    try:
        images = convert_from_path(pdf_path)  # Convert PDF pages to images
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image)  # Extract text from each image
        return text
    except Exception as e:
        print(f"Error processing PDF as images: {e}")
        return ""


# Step 4: Combined function to handle both text-based and image-based PDFs
def extract_text(pdf_path):
    if not is_poppler_installed():
        print("Poppler is not installed or not in PATH. Please install it.")
        return ""

    try:
        text = extract_text_from_pdf(pdf_path)
        if text.strip():
            return text
        else:
            raise ValueError("No text found in PDF, attempting OCR...")
    except Exception as e:
        print(f"Error: {e}")
        return extract_text_from_images(pdf_path)


# Step 5: Load the tokenizer for LLaMA
def load_tokenizer(model_name="meta-llama/Llama-3.2-3B-Instruct"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None


# Step 6: Tokenize text and prepare chunks
def tokenize_text(text, tokenizer, max_length=2048):
    """
    Tokenizes text and splits it into manageable chunks of `max_length`.
    Returns decoded string chunks.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk_tokens = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
    return chunks



# Step 7: Preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.upper()  # Convert to uppercase
    return text


# Step 8: Query the LLaMA model
def query_llama(api_key, input_text):
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": input_text, "parameters": {"max_length": 200}}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return None


# Step 9: Find relevant chunk for a question
def find_relevant_chunk(chunks, question):
    """
    Finds the chunk of text most relevant to the question.
    """
    for chunk in chunks:
        if question.lower() in chunk.lower():
            return chunk
    return chunks[0]  # Default to the first chunk if no match is found



# Step 10: Answer a question
def answer_question(api_key, chunks, question, tokenizer):
    """
    Finds the relevant chunk and queries the LLaMA model for an answer.
    """
    context = find_relevant_chunk(chunks, question)  # No need to decode again; already string
    response = query_llama(api_key, f"Context: {context}\nQuestion: {question}")
    if response:
        return response[0]["generated_text"]
    return "Unable to generate an answer."

# Step 11: Browse for a PDF file
def browse():
    # Create a Tkinter root window (hidden)
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open a file dialog to select a PDF file
    file_path = filedialog.askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF files", "*.pdf")],  # Filter for PDF files only
    )

    if file_path:
        print(f"Selected file: {file_path}")
        return file_path
    else:
        print("No file selected.")
        return None


# Main Program
if __name__ == "__main__":
    pdf_path = browse()
    api_key = "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"

    if not os.path.exists(pdf_path):
        print("PDF file not found.")
    else:
        # Extract and preprocess text
        extracted_text = preprocess_text(extract_text(pdf_path))
        if not extracted_text.strip():
            print("Failed to extract text.")
            exit()

        # Load tokenizer
        tokenizer = load_tokenizer()
        if not tokenizer:
            print("Failed to load tokenizer.")
            exit()

        # Tokenize text into chunks
        chunks = tokenize_text(extracted_text, tokenizer)

        # Interactive Question Answering
        while True:
            question = input("\nAsk a question (or type 'exit' to quit): ")
            if question.lower() == "exit":
                break
            answer = answer_question(api_key, chunks, question, tokenizer)
            answer=answer.split(sep=question)
            print(f"Answer: {answer[1]}")