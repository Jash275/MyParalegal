import os
from flask import Flask, render_template, request, redirect, url_for, flash
from pdf2image import convert_from_path
import pytesseract
from transformers import BartTokenizer, BartForConditionalGeneration

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Set the path for Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Function to summarize text using BART
def summarize_text(text):
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to extract text from PDF using Tesseract
def extract_text_from_pdf(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return str(e)

# Home route to display the form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and summarization
@app.route('/upload', methods=['POST'])
def upload():
    if 'pdf' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['pdf']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        # Save the uploaded PDF to a temporary location
        pdf_path = os.path.join('uploads', file.filename)
        file.save(pdf_path)

        # Extract text from PDF
        extracted_text = extract_text_from_pdf(pdf_path)

        if extracted_text.strip():
            # Summarize the extracted text
            summary = summarize_text(extracted_text)
        else:
            summary = "No text extracted from the PDF."

        # Remove the uploaded file after processing
        os.remove(pdf_path)

        # Render the HTML template with the summary
        return render_template('index.html', summary=summary)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
