import re
from PyPDF2 import PdfReader
from langchain_core.documents import Document

def parse_pdf_with_pypdf(pdf_path):
    """Extracts and cleans text from a PDF, converting each page into a Document."""
    reader = PdfReader(pdf_path)
    documents = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            cleaned_text = clean_text(text)
            documents.append(Document(page_content=cleaned_text, metadata={"page_number": page_num + 1}))
    return documents


def clean_text(text):
    # Remove excessive newlines, spaces, and HTML artifacts
    text = re.sub(r'\n+', '\n', text)  
    text = re.sub(r'\s{2,}', ' ', text)  
    return text.strip()
