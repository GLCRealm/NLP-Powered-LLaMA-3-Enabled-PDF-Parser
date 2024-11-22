import tkinter as tk
from tkinter import filedialog, Text, Scrollbar
from tkinter import ttk
import start  # Importing your backend file

pdf_path = None

# Function to browse and select a PDF
def browse_file():
    global pdf_path
    pdf_path = filedialog.askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF files", "*.pdf")],
    )
    if pdf_path:
        pdf_name_label.config(text=f"PDF Name: {pdf_path.split('/')[-1]}")
        # Call your backend to load and process the PDF
        try:
            start.extract_text(pdf_path)  # Replace 'load_pdf' with the actual function in start.py
            preview_label.config(text="PDF successfully loaded!")
        except Exception as e:
            preview_label.config(text=f"Error loading PDF: {str(e)}")
    else:
        preview_label.config(text="No PDF selected!")

# Function to handle question submission
def submit_question():
    global pdf_path
    question = question_entry.get()
    if not pdf_path:
        chat_text.insert(tk.END, "Error: No PDF uploaded. Please upload a PDF first.\n")
        return

    if question.strip():
        chat_text.insert(tk.END, f"User: {question}\n")
        try:
            # Use your backend's function to get the answer
            api_key = "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"
            print(pdf_path)
            extracted_text = start.preprocess_text(start.extract_text(pdf_path))
            tokenizer = start.load_tokenizer()
            chunks = start.tokenize_text(extracted_text, tokenizer)

            answer = start.answer_question(api_key, chunks, question, tokenizer)  # Replace with the actual function in start.py
            answer = answer.split(sep=question)

            chat_text.insert(tk.END, f"Model: {answer[1]}\n")
        except Exception as e:
            chat_text.insert(tk.END, f"Error: {str(e)}\n")
        chat_text.see(tk.END)  # Scroll to the end
        question_entry.delete(0, tk.END)

# Initialize the GUI
root = tk.Tk()
root.title("Book Assistance GUI")
root.geometry("800x600")
root.configure(bg="#2b2b2b")  # Dark theme background

# PDF Name section
pdf_name_label = tk.Label(root, text="No PDF uploaded", fg="white", bg="#2b2b2b", font=("Helvetica", 12))
pdf_name_label.pack(pady=10)

browse_button = ttk.Button(root, text="Upload PDF", command=browse_file)
browse_button.pack(pady=5)

# Chat Section
chat_frame = tk.Frame(root, bg="#2b2b2b")
chat_frame.pack(pady=10, fill=tk.BOTH, expand=True)

chat_text = Text(chat_frame, wrap=tk.WORD, bg="#1e1e1e", fg="white", font=("Helvetica", 10), state=tk.NORMAL)
chat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

chat_scrollbar = Scrollbar(chat_frame, command=chat_text.yview)
chat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
chat_text.config(yscrollcommand=chat_scrollbar.set)

# PDF Preview Section
preview_frame = tk.Frame(root, bg="#2b2b2b")
preview_frame.pack(pady=10, fill=tk.BOTH, expand=True)

preview_label = tk.Label(preview_frame, text="Uploaded PDF Preview (Placeholder)", fg="white", bg="#2b2b2b", font=("Helvetica", 12))
preview_label.pack()

# Question Entry Section
question_frame = tk.Frame(root, bg="#2b2b2b")
question_frame.pack(pady=10)

question_entry = tk.Entry(question_frame, width=50, font=("Helvetica", 12), bg="#1e1e1e", fg="white")
question_entry.grid(row=0, column=0, padx=5, pady=5)

submit_button = ttk.Button(question_frame, text="Submit", command=submit_question)
submit_button.grid(row=0, column=1, padx=5, pady=5)

# Start the GUI loop
root.mainloop()
