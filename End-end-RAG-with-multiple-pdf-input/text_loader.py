import pdfplumber
from logger import get_logger

logger = get_logger("TextLoader")

def extract_text_chunks(pdf_path):
    
    chunks = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    chunks.append({
                        "text": text.strip(),
                        "page": page_number,
                        "type": "text"
                    })
    except Exception as e:
        logger.error(f"Error extracting text: {e}")

    logger.info(f"Extracted text from {len(chunks)} pages.")
    return chunks
