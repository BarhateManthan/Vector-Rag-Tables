import pandas as pd
import pdfplumber
from logger import get_logger

logger = get_logger("TableLoader")

def extract_table_chunks(pdf_path):
 
    chunks = []
    total_tables = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"Number of pages in the PDF: {len(pdf.pages)}")

            for page_number, page in enumerate(pdf.pages, start=1):
                page_tables = page.extract_tables()
                total_tables += len(page_tables)

                for table_idx, table in enumerate(page_tables, start=1):
                    df = pd.DataFrame(table[1:], columns=table[0])
                    if not df.empty:
                        markdown = df.to_markdown(index=False)
                        chunks.append({
                            "text": markdown,
                            "page": page_number,
                            "type": "table"
                        })

        logger.info(f"Extracted {len(chunks)} tables from {pdf_path}")
    except Exception as e:
        logger.error(f"Error extracting tables: {e}")

    return chunks, total_tables
