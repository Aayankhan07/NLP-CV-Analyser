import pdfplumber
import docx
import io
import re
import os
import pytesseract
from PIL import Image

def clean_text(text: str) -> str:
    """
    Cleans the extracted text by lowercasing and removing excessive whitespace.
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace multiple spaces, newlines, and tabs with a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def parse_pdf(file_obj) -> str:
    """
    Parses a PDF file object and extracts text line-by-line using pdfplumber.
    """
    text = ""
    try:
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return clean_text(text)

def parse_docx(file_obj) -> str:
    """
    Parses a DOCX file object and extracts text using python-docx.
    """
    text = ""
    try:
        doc = docx.Document(file_obj)
        for para in doc.paragraphs:
            if para.text:
                text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX: {e}")
    return clean_text(text)

def parse_image(file_obj) -> str:
    """
    Parses an image file and extracts text using Tesseract OCR.
    """
    text = ""
    try:
        # Common Windows installation path for Tesseract
        tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
        image = Image.open(file_obj)
        text = pytesseract.image_to_string(image)
    except Exception as e:
        error_msg = f"Tesseract Error: {e}"
        print(error_msg)
        # If it's a TesseractNotFoundError, we can give a more specific hint
        if "tesseract is not installed" in str(e).lower():
            return "ERROR: Tesseract OCR is not installed or not in PATH."
        return f"ERROR: {error_msg}"
        
    return clean_text(text)

def parse_file(file_obj, filename: str) -> str:
    """
    Determines the file type and uses the appropriate parser.
    """
    filename = filename.lower()
    if filename.endswith(".pdf"):
        return parse_pdf(file_obj)
    elif filename.endswith(".docx"):
        return parse_docx(file_obj)
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        return parse_image(file_obj)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF, DOCX, or Image file.")
